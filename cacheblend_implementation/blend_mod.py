from vllm import LLM, SamplingParams
import torch
import json
from transformers import AutoTokenizer
import numpy as np

def move_to_cpu_numpy(obj):
    if isinstance(obj, torch.Tensor):
        obj = obj.cpu()
        if obj.dtype == torch.bfloat16:
            obj = obj.to(torch.float32)  # Convert bfloat16 -> float32
        return obj.numpy()
    elif isinstance(obj, dict):
        return {k: move_to_cpu_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cpu_numpy(v) for v in obj)
    else:
        return obj

    
output_dict = {
    'old_kv_cache': [],
    'recomputed_kv_cache_first_layer': [],
    'recomputed_kv_cache_subsequent_layers': [],
    'important_tokens': [],
    'layer_outputs': [],
    'rotary_embeddings': [],
}

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.9,
          #tokenizer=tokenizer,
          )
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
llm.set_tokenizer(tokenizer)

#TODO (Jiayi): fix last len


for sample_idx in range(1,11):
    f = open(f"inputs/{sample_idx}.json")
    ex = json.load(f)
    chunk_num = ex['chunk_num']
    doc_prompts = [ex[f'{i}'] for i in range(chunk_num)]
    q_prompt = ex['query']
    doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
    q_ids = tokenizer.encode(q_prompt)[1:]


    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, max_tokens=1)

    # Create an tokenizer and LLM.
    cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
    cache_fuse_metadata['collect'] = False
    cache_fuse_metadata['check'] = False

    s_start_full = [733, 4138, 28793]
    s_start_len = len(s_start_full) + 1

    #s_start = [518, 25580, 29962]
    s_start = []
    s_start_1_len = len(s_start) + 1

    s_end = [518, 29914, 25580, 29962]
    s_end = [733, 28748, 16289, 28793]
    s_end_len = len(s_end)
    old_kvs = []

    doc_chunk_ids = [s_start+chunk_ids for chunk_ids in doc_chunk_ids]
    doc_chunk_ids = [s_start_full] + doc_chunk_ids
    doc_chunk_ids = doc_chunk_ids + [s_start+q_ids+s_end]

    last_len = len([q_ids+s_end])

    cache_fuse_metadata['collect'] = True
    cache_fuse_metadata["check"] = False

    num_layer = 32
    chunk_past_key_values = []
    
    # Concatenate old KVs
    for i in range(len(doc_chunk_ids)):
        prompts = [tokenizer.decode(doc_chunk_ids[i])]
        llm.generate(prompts, sampling_params)
        
        llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
        for j in range(num_layer):
            past_key_values = llm_layers[j].self_attn.hack_kv
            if i == 0:
                temp_k = past_key_values[0][:s_start_len].clone() # do not chage with s_start_1
                temp_v = past_key_values[1][:s_start_len].clone()
            else:
                temp_k = past_key_values[0][s_start_1_len:len(doc_chunk_ids[i])+1].clone()
                temp_v = past_key_values[1][s_start_1_len:len(doc_chunk_ids[i])+1].clone()    

            if i == 0:
                chunk_past_key_values.append([temp_k, temp_v])
            else:
                #pdb.set_trace()
                chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0],temp_k), dim=0)
                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1],temp_v), dim=0)
        #print(temp_k.shape[0])
        llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = chunk_past_key_values
        
    input_ids = []

    for i in range(len(doc_chunk_ids)):
        if i == 0:
            temp_ids = doc_chunk_ids[i]
        else:
            temp_ids = doc_chunk_ids[i][s_start_1_len-1:]
        input_ids += temp_ids
        
    input_prompt = tokenizer.decode(input_ids)
 

    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    cache_fuse_metadata["check"] = True
    cache_fuse_metadata['collect'] = True
    cache_fuse_metadata['suffix_len'] = last_len
    output = llm.generate([input_prompt], sampling_params)
    print(f"Cached generation: {output[0].outputs[0].text}")
    print(f"TTFT with cache: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
    
    # Save data in npz format
    output_dict['old_kv_cache'] = [kv for kv in chunk_past_key_values]  # Store old KVs
    
    # Capture the important tokens (token id and position)
    imp_tokens = []  # Placeholder: update based on actual important token logic
    imp_positions = []  # Placeholder: update based on actual positions
    for layer_idx in range(num_layer):
        imp_tokens.append(llm_layers[layer_idx].self_attn.imp_tokens)
        imp_positions.append(llm_layers[layer_idx].self_attn.imp_indices)
    output_dict['important_tokens'] = {'token_ids': imp_tokens, 'positions': imp_positions}
    
    # Store outputs after each attention layer
    output_dict['layer_outputs'] = [kv[0] for kv in chunk_past_key_values]
    
    # Capture rotary embeddings
    rotary_embs = [llm_layers[j].self_attn.rotary_emb for j in range(num_layer)]  # Collect rotary embeddings
    output_dict['rotary_embeddings'] = rotary_embs

    # Save everything as an npz file
    cpu_output_dict = move_to_cpu_numpy(output_dict)
    np.savez(f"output_data_{sample_idx}.npz", **cpu_output_dict)

    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    cache_fuse_metadata["check"] = False
    cache_fuse_metadata['collect'] = False
    output = llm.generate([input_prompt], sampling_params)
    print(f"Normal generation: {output[0].outputs[0].text}")
    print(f"TTFT with full prefill: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
    print("------------")