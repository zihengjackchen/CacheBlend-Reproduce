"""Attention layer with xFormers and PagedAttention."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (AttentionBias,
                                         BlockDiagonalCausalMask,
                                         LowerTriangularMaskWithTensorBias)

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataPerStage)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
from vllm.logger import init_logger

logger = init_logger(__name__)


class XFormersBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["XFormersImpl"]:
        return XFormersImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "XFormersMetadata":
        return XFormersMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class XFormersMetadata(AttentionMetadataPerStage, PagedAttentionMetadata):
    """Metadata for XFormersbackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    # (batch_size,). The prompt length per sequence. None if it is a decoding.
    prompt_lens: Optional[List[int]]
    # prompt_lens stored as a tensor.
    prompt_lens_tensor: Optional[torch.Tensor]

    # NOTE(sang): Definition of context_len, subquery_len, and seqlen.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seqlen ----------------------|
    #                                   |- subquery_len -|

    # WARNING(sang): context_len has different definition depending on if it is
    # prefill vs decoding. When it is prefill, it doesn't include new tokens.
    # When it is for decoding, it includes a new token.

    # Maximum subquery length in the batch.
    max_subquery_len: Optional[int]
    # FIXME: It is for flash attn.
    # Maximum prompt length in the batch.
    max_prompt_len: Optional[int]
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    subquery_start_loc: Optional[torch.Tensor]
    # FIXME: It is for flash attn.
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[AttentionBias]] = None


class XFormersImpl(AttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|	
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:	
    |<----------------- num_decode_tokens ------------------>|	
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        suppored_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata[XFormersMetadata],
        kv_scale: float,
        status: int,
        cache_fuse_metadata: dict,
        old_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Forward pass with xFormers and PagedAttention, supporting CacheBlend selective recompute.

        Args:
            query: (num_tokens, num_heads * head_size)
            key: (num_tokens, num_kv_heads * head_size)
            value: (num_tokens, num_kv_heads * head_size)
            kv_cache: optional cache to write updated keys/values
            attn_metadata: metadata object containing prefill/decode information
            kv_scale: scaling factor for KV cache
            status:
                0 = Normal inference (no CacheBlend)
                1 = Check Layer (select important tokens and recompute)
                2 = Subsequent Layers (recompute only selected important tokens)
            cache_fuse_metadata: dictionary to pass CacheBlend metadata across layers
            old_kv: old key and value tensors from the precomputed cache (only needed for status 1 and 2)

        Returns:
            Tensor of shape (num_tokens, num_heads * head_size)
        """
        num_tokens, hidden_size = query.shape
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        # Prepare old KV cache if CacheBlend is active
        if status in [1, 2]:
            old_key, old_value = old_kv
            old_key = old_key.view(-1, self.num_kv_heads, self.head_size)
            old_value = old_value.view(-1, self.num_kv_heads, self.head_size)

        if status == 1:
            # --- Check Layer: Select important tokens to recompute ---
            suffix_len = cache_fuse_metadata['suffix_len']
            total_len = value.shape[0]

            suffix_indices = torch.arange(total_len - suffix_len, total_len, device=value.device)

            # Compute L2 distance between new value and old value for non-suffix tokens
            value_diff = torch.sum((value[:-suffix_len] - old_value[:-suffix_len]) ** 2, dim=[1, 2])

            # Select top tokens with largest changes
            num_top_tokens = int((total_len - suffix_len) * cache_fuse_metadata["recomp_ratio"])
            top_diff_indices = torch.topk(value_diff, k=num_top_tokens).indices
            top_diff_indices = torch.sort(top_diff_indices).values

            # Merge top-diff tokens and all suffix tokens
            important_indices = torch.cat([top_diff_indices, suffix_indices])

            cache_fuse_metadata["imp_indices"] = important_indices

            # Set attention mask for partial tokens
            attn_bias = _make_partial_bias_gqa(cache_fuse_metadata, query.device, self.num_kv_heads, self.num_queries_per_kv)
            cache_fuse_metadata["attn_bias"] = attn_bias

            # Clear old attention bias
            attn_metadata.prefill_metadata.attn_bias = None

        cache_fuse_metadata["kv_cache_dtype"] = value.dtype

        if status == 2:
            # --- Update old KV by replacing important tokens ---
            important_indices = cache_fuse_metadata["imp_indices"]
            old_key[important_indices] = key
            old_value[important_indices] = value
            key = old_key
            value = old_value

        # Write key and value to KV cache if provided
        if kv_cache is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            PagedAttention.write_to_paged_cache(
                key, value,
                key_cache, value_cache,
                attn_metadata.slot_mapping,
                attn_metadata.kv_cache_dtype,
                kv_scale
            )

        # Slice query, key, value for prefill and decode
        if status in [1, 2]:
            query = query[cache_fuse_metadata["imp_indices"]]
            num_prefill_tokens = len(cache_fuse_metadata["imp_indices"])
            num_decode_tokens = 0
            output = torch.empty((num_prefill_tokens, self.num_heads, self.head_size), dtype=query.dtype, device=query.device)
            decode_query = None
        else:
            num_prefill_tokens = attn_metadata.num_prefill_tokens
            num_decode_tokens = attn_metadata.num_decode_tokens

            decode_query = query[num_prefill_tokens:]
            query = query[:num_prefill_tokens]
            key = key[:num_prefill_tokens]
            value = value[:num_prefill_tokens]

            output = torch.empty((num_prefill_tokens + num_decode_tokens, self.num_heads, self.head_size), dtype=query.dtype, device=query.device)

        # --- Prefill Attention ---
        if prefill_meta := attn_metadata.prefill_metadata:
            if kv_cache is None or prefill_meta.block_tables.numel() == 0:
                out = self._run_memory_efficient_xformers_forward(
                    query, key, value,
                    prefill_meta,
                    status,
                    cache_fuse_metadata
                )
                output[:out.shape[0]] = out
            else:
                out = PagedAttention.forward_prefix(
                    query, key, value,
                    key_cache, value_cache,
                    prefill_meta.block_tables,
                    prefill_meta.subquery_start_loc,
                    prefill_meta.prompt_lens_tensor,
                    prefill_meta.context_lens,
                    prefill_meta.max_subquery_len,
                    self.alibi_slopes,
                )
                output[:num_prefill_tokens] = out

        # --- Decode Attention (Only for normal state) ---
        if decode_meta := attn_metadata.decode_metadata and status == 0:
            output[num_prefill_tokens:] = PagedAttention.forward_decode(
                decode_query,
                key_cache,
                value_cache,
                decode_meta.block_tables,
                decode_meta.context_lens,
                decode_meta.max_context_len,
                attn_metadata.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                kv_scale,
            )

        # Reshape and return
        return output.view(-1, self.num_heads * self.head_size)


    def _run_memory_efficient_xformers_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: XFormersMetadata,
        status: int,
        cache_fuse_metadata: dict,
    ) -> torch.Tensor:
        """
        Run memory-efficient attention using xFormers, supporting CacheBlend for selective tokens.

        Args:
            query: (num_tokens, num_heads, head_size)
            key: (num_tokens, num_kv_heads, head_size)
            value: (num_tokens, num_kv_heads, head_size)
            attn_metadata: metadata object
            status: CacheBlend status (0/1/2)
            cache_fuse_metadata: CacheBlend metadata
        Returns:
            Tensor (num_tokens, num_heads, head_size)
        """
        assert attn_metadata.prompt_lens is not None

        # Prepare for GQA/MQA if num_heads != num_kv_heads
        if self.num_kv_heads != self.num_heads:
            query = query.view(query.shape[0], self.num_kv_heads, self.num_queries_per_kv, query.shape[-1])
            key = key.unsqueeze(2).expand(-1, self.num_kv_heads, self.num_queries_per_kv, -1)
            value = value.unsqueeze(2).expand(-1, self.num_kv_heads, self.num_queries_per_kv, -1)

        # Setup attention bias
        if attn_metadata.attn_bias is None:
            if status == 1:
                # Use partial mask for important tokens
                attn_metadata.attn_bias = [cache_fuse_metadata["attn_bias"]]
            else:
                if self.alibi_slopes is None:
                    attn_bias = BlockDiagonalCausalMask.from_seqlens(attn_metadata.prompt_lens)
                    if self.sliding_window is not None:
                        attn_bias = attn_bias.make_local_attention(self.sliding_window)
                    attn_metadata.attn_bias = [attn_bias]
                else:
                    attn_metadata.attn_bias = _make_alibi_bias(
                        self.alibi_slopes,
                        self.num_kv_heads,
                        query.dtype,
                        attn_metadata.prompt_lens
                    )

        # Perform attention
        output = xops.memory_efficient_attention(
            query,
            key,
            value,
            attn_bias=attn_metadata.attn_bias[0]
        )
        return output



def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    dtype: torch.dtype,
    prompt_lens: List[int],
) -> LowerTriangularMaskWithTensorBias:
    attn_biases = []
    for prompt_len in prompt_lens:
        bias = torch.arange(prompt_len, dtype=dtype)
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(prompt_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        # Calculate a matrix where each element represents ith element- jth
        # element.
        bias = bias[None, :] - bias[:, None]

        padded_len = (prompt_len + 7) // 8 * 8
        num_heads = alibi_slopes.shape[0]
        bias = torch.empty(
            1,  # batch size
            num_heads,
            prompt_len,
            padded_len,
            device=alibi_slopes.device,
            dtype=dtype,
        )[:, :, :, :prompt_len].copy_(bias)
        bias.mul_(alibi_slopes[:, None, None])
        if num_heads != num_kv_heads:
            bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
        attn_biases.append(LowerTriangularMaskWithTensorBias(bias))

    return attn_biases

def _make_partial_bias_gqa(cache_fuse_metadata, 
                       device,
                       num_kv_heads,
                       num_queries_per_kv,):
    seq_len = cache_fuse_metadata['org_seq_len']
    padded_len = (seq_len + 7) // 8 * 8
    dtype = cache_fuse_metadata['kv_cache_dtype']
    imp_indices = cache_fuse_metadata['imp_indices']
    attn_mask = torch.triu(torch.ones(padded_len,
                                      padded_len,
                                      dtype=dtype,
                                      device=device),
                           diagonal=1)
    #FIXME(Jiayi): The first 1 (bsz) is a hack
    attn_mask = (attn_mask * torch.finfo(dtype).min).view(1, 
                                                          1, 1, padded_len, padded_len) #FIXME(Jiayi): Now only focus on bsz=1
    attn_mask = attn_mask[:,:,:,imp_indices]
    attn_mask = attn_mask.expand(1,
                                 num_kv_heads,num_queries_per_kv,-1,-1)
    #import pdb
    #pdb.set_trace()
    attn_mask_padded = torch.empty(
        1,
        num_kv_heads,
        num_queries_per_kv,
        len(imp_indices),
        padded_len,
        device=device,
        dtype=dtype,
    ).copy_(attn_mask)[:, :, :, :, :seq_len]
    #attn_mask_padded = LowerTriangularMaskWithTensorBias(attn_mask_padded)
    return attn_mask_padded

def _fetch_maetrailized_mask_gqa(q_len,num_kv_heads,num_queries_per_kv,device,dtype):
    seq_len = q_len
    padded_len = (seq_len + 7) // 8 * 8
    attn_mask = torch.triu(torch.ones(seq_len,
                                      padded_len,
                                      dtype=dtype,
                                      device=device),
                           diagonal=1)
    #FIXME(Jiayi): The first 1 (bsz) is a hack
    attn_mask = (attn_mask * torch.finfo(dtype).min).view(#1,
                                                          1, 1, seq_len, padded_len) #FIXME(Jiayi): Now only focus on bsz=1
    attn_mask = attn_mask.expand(#1,
                                 num_kv_heads, num_queries_per_kv,-1,-1)
    
    attn_mask_padded = torch.empty(
        #1,
        num_kv_heads,
        num_queries_per_kv,
        seq_len,
        padded_len,
        device=device,
        dtype=dtype,
    ).copy_(attn_mask)[#:, 
                       :, :, :, :seq_len]
    #attn_mask_padded = LowerTriangularMaskWithTensorBias(attn_mask_padded)
    return attn_mask_padded