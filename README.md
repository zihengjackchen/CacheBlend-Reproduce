# Compress & Blend: Unified Framework for Efficient KV Cache Optimization in RAG

This repository contains the code, scripts, and experimental results for the project **Compress & Blend**, which explores strategies for recomputation, eviction, and compression of key-value (KV) caches to enable scalable and efficient retrieval-augmented generation (RAG) with large language models (LLMs). The framework extends and evaluates methods such as **CacheBlend**, **EPIC**, and **xKV** in a unified evaluation pipeline.

---

## 📁 Directory Overview

### `CacheBlend-Reproduce/output/`

Contains experimental outputs from different KV cache recomputation strategies under the CacheBlend framework.

Each subfolder corresponds to a specific token selection or recomputation method:

- **`select_full_recompute/`**  
  Baseline results from fully recomputing the KV cache (100% recomputation).

- **`select_imp_from_v_baseline/`**  
  CacheBlend default: selects important tokens based on deviation in **value (V)** vectors.

- **`select_imp_from_k/`**  
  Selects tokens based on **key (K)** vector deviations instead of V.

- **`select_imp_no_selection/`**  
  No recomputation; uses all precomputed KV entries as-is (fastest but least accurate).

- **`select_imp_2nd_layer/`**, **`select_imp_3rd_layer/`**  
  Token importance selection happens at deeper transformer layers, testing if later-layer deviation improves results.

- **`select_imp_4_20_layer/`**, **`select_imp_8_24_layer/`**, **`select_imp_12_28_layer/`**, **`select_imp_16_32_layer/`**  
  Recomputes **mid-ranked tokens** instead of top 16% to evaluate if less obvious tokens contribute to performance.

Each folder contains output samples, TTFT metrics, and token selection consistency comparisons.

---

### `CacheBlend-Reproduce/testing_blend_selection/`

Scripts and summary tables for evaluating and comparing the various strategies.

#### Scripts:
- **`blend_extract_imp.py`**  
  Extracts important token indices from attention deviation metrics (V or K norms).

- **`blend_diff_recomptue.py`**  
  Compares model outputs across recomputation strategies to assess semantic deviation and correctness.

- **`blend_diff_selections.py`**  
  Analyzes overlap and difference in token selection across methods (e.g., EPIC vs CacheBlend).

- **`method_comparison.py`**  
  Runs benchmarks, logs Time To First Token (TTFT), accuracy, and consistency metrics for each method.

#### Data:
- **`method_comparison.txt`**  
  Log or notes from head-to-head comparisons of recomputation strategies.

- **`summary_table.csv`**  
  Tabulates output correctness, TTFT, and consistency across all recomputation methods.

- **`summary_table_recompute.csv`**  
  A focused summary of recomputation-related metrics (e.g., recomputed token ratios, TTFT).

---

## 📄 Project Summary

The project evaluates trade-offs between:
- **Compute savings** via selective token recomputation (CacheBlend, EPIC).
- **Memory savings** via token eviction (Attention Sink, H2O).
- **Compression** through xKV-style cross-layer SVD.

It introduces a **Compress-and-Blend** workflow, enabling flexible tuning between speed, memory, and accuracy. The framework also proposes a novel **token categorization** scheme to distinguish intrinsic, relational, and dummy tokens.

Key results and comparisons are included in our [report (PDF)](./report.pdf).

---
Here's a section you can add to your `README.md` to introduce the results table:

---

## 📊 Experimental Results
The table below summarizes the output quality and average Time To First Token (TTFT) across various token recomputation strategies on a 10-query benchmark. Each row corresponds to a sample question, and each column shows the model output and TTFT under a specific method, including:

* **Full Recompute**: Baseline with 100% recomputation.
* **CacheBlend Variants**: Select tokens based on deviation in value (V) or key (K), or across different rank ranges.
* **Layer-wise Selection**: Importance estimated at deeper transformer layers (2nd or 3rd).
* **No Selection**: Reuse all cached tokens with no recomputation (fastest but often least accurate).

This comparison illustrates the trade-offs between latency and output fidelity, reinforcing the need for context-aware and hybrid recomputation strategies. Lower TTFT typically corresponds to more aggressive approximations, which may impact answer quality.


| Sample           | Question                                                                                                     | select_full_recompute                                                   | select_imp_2nd_token                                                         | select_imp_3rd_layer                                                         | select_imp_4_20_layer                                                             | select_imp_8_24_layer                                                          | select_imp_12_28_layer                                                        | select_imp_16_32_layer                                                       | select_imp_from_k                                                               | select_imp_from_v_baseline                                                   | select_imp_no_selection                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| output_sample_1  | Where did the performer of song I'll Say It graduate from?                                                   | output_text: Carnegie Mellon University.<br>ttft_avg: 0.263<br><br>     | output_text: Carnegie Mellon University.<br>ttft_avg: 0.083<br><br><br>      | output_text: Graduated from West Berlin.<br>ttft_avg: 0.157<br><br><br>      | output_text: Carnegie Mellon University<br>ttft_avg: 0.076<br><br><br>            | output_text: Carnegie Mellon University<br>ttft_avg: 0.074<br><br><br>         | output_text: Carnegie Mellon University<br>ttft_avg: 0.069<br><br><br>        | output_text: Graduated from Carnegie Mellon.<br>ttft_avg: 0.081<br><br><br>  | output_text: Carnegie Mellon University.<br>ttft_avg: 0.074<br><br><br>         | output_text: Carnegie Mellon University.<br>ttft_avg: 0.074<br><br><br>      | output_text: Graduated from Carnegie Mellon.<br>ttft_avg: 0.047<br><br><br>         |
| output_sample_2  | Which film has the director who was born first, Hell Up In Harlem or The Soviet Story?                       | output_text: The Soviet Story (older)<br>ttft_avg: 0.221<br><br>        | output_text: The Soviet Story was made later.<br>ttft_avg: 0.070<br><br><br> | output_text: Hell Up In Harlem. (Lee Thompson<br>ttft_avg: 0.144<br><br><br> | output_text: The Soviet Story was made first.<br>ttft_avg: 0.065<br><br><br>      | output_text: Hell Up In Harlem (1973<br>ttft_avg: 0.065<br><br><br>            | output_text: Hell Up In Harlem (older)<br>ttft_avg: 0.065<br><br><br>         | output_text: Hell Up In Harlem (older)<br>ttft_avg: 0.065<br><br><br>        | output_text: The Soviet Story was made first.<br>ttft_avg: 0.065<br><br><br>    | output_text: The Soviet Story was made first.<br>ttft_avg: 0.065<br><br><br> | output_text: The Soviet Story was made first.<br>ttft_avg: 0.028<br><br><br>        |
| output_sample_3  | Who was born first, Cipriano Castro or Damir Nikšić?                                                         | output_text: Cipriano Castro was born first.<br>ttft_avg: 0.220<br><br> | output_text: Cipriano Castro was born first.<br>ttft_avg: 0.069<br><br><br>  | output_text: Cipriano Castro was born first.<br>ttft_avg: 0.144<br><br><br>  | output_text: Cipriano Castro born before Damir<br>ttft_avg: 0.065<br><br><br>     | output_text: Cipriano Castro born before Damir<br>ttft_avg: 0.064<br><br><br>  | output_text: Cipriano Castro born before Damir<br>ttft_avg: 0.065<br><br><br> | output_text: Cipriano Castro born before Nikš<br>ttft_avg: 0.065<br><br><br> | output_text: Cipriano Castro was born before Dam<br>ttft_avg: 0.064<br><br><br> | output_text: Cipriano Castro was born first.<br>ttft_avg: 0.064<br><br><br>  | output_text: Cipriano Castro born before Nikš<br>ttft_avg: 0.028<br><br><br>        |
| output_sample_4  | Which song came out first, Joel The Lump Of Coal or Jugband Blues?                                           | output_text: Jugband Blues came out first.<br>ttft_avg: 0.262<br><br>   | output_text: "Jugband Blues" came out first.<br>ttft_avg: 0.074<br><br><br>  | output_text: Jugband Blues came first.<br>ttft_avg: 0.157<br><br><br>        | output_text: Joel The Lump Of Coal came out first.<br>ttft_avg: 0.068<br><br><br> | output_text: Jugband Blues came out first.<br>ttft_avg: 0.068<br><br><br>      | output_text: Jugband Blues came first.<br>ttft_avg: 0.068<br><br><br>         | output_text: Jugband Blues came first.<br>ttft_avg: 0.068<br><br><br>        | output_text: Jugband Blues came out first.<br>ttft_avg: 0.068<br><br><br>       | output_text: Jugband Blues came out first.<br>ttft_avg: 0.068<br><br><br>    | output_text: Jugband Blues came first.<br>ttft_avg: 0.030<br><br><br>               |
| output_sample_5  | What is the place of birth of Ratna Malla's father?                                                          | output_text: Bulacan, Philippines.<br>ttft_avg: 0.255<br><br>           | output_text: Bulacan, Philippines.<br>ttft_avg: 0.073<br><br><br>            | output_text: Ducal house of Bourbon.<br>ttft_avg: 0.152<br><br><br>          | output_text: Bulacan, Philippines.<br>ttft_avg: 0.068<br><br><br>                 | output_text: Bulacan province.<br>ttft_avg: 0.067<br><br><br>                  | output_text: Unknown, mentioned as "village"<br>ttft_avg: 0.067<br><br><br>   | output_text: Unknown, mentioned as "village"<br>ttft_avg: 0.067<br><br><br>  | output_text: Bulacan, Philippines<br>ttft_avg: 0.067<br><br><br>                | output_text: Bulacan, Philippines.<br>ttft_avg: 0.067<br><br><br>            | output_text: Unknown, mentioned as "a village in bul<br>ttft_avg: 0.030<br><br><br> |
| output_sample_6  | Who is Archibald Acheson, 4Th Earl Of Gosford's paternal grandfather?                                        | output_text: Arthur Acheson, 1st Earl of<br>ttft_avg: 0.256<br><br>     | output_text: Arthur Acheson, 1st Earl of<br>ttft_avg: 0.073<br><br><br>      | output_text: King Robert II of Scotland.<br>ttft_avg: 0.156<br><br><br>      | output_text: King Robert II of Scotland<br>ttft_avg: 0.068<br><br><br>            | output_text: Archibald Acheson, 2nd<br>ttft_avg: 0.068<br><br><br>             | output_text: Archibald Acheson, 2nd<br>ttft_avg: 0.068<br><br><br>            | output_text: Archibald Acheson, 2nd<br>ttft_avg: 0.068<br><br><br>           | output_text: King Robert II of Scotland.<br>ttft_avg: 0.068<br><br><br>         | output_text: Arthur Acheson, 1st Earl of<br>ttft_avg: 0.067<br><br><br>      | output_text: Archibald Acheson, 2nd<br>ttft_avg: 0.030<br><br><br>                  |
| output_sample_7  | Where did the performer of song Fantasy (George Michael Song) die?                                           | output_text: At home in Goring-on-Thames<br>ttft_avg: 0.263<br><br>     | output_text: Goring-on-Thames.<br>ttft_avg: 0.074<br><br><br>                | output_text: Goring-on-Thames, Oxfordshire<br>ttft_avg: 0.159<br><br><br>    | output_text: Goring-on-Thames, Oxfordshire<br>ttft_avg: 0.069<br><br><br>         | output_text: Goring-on-Thames, Oxfordshire<br>ttft_avg: 0.069<br><br><br>      | output_text: Goring-on-Thames, Oxfordshire<br>ttft_avg: 0.068<br><br><br>     | output_text: In Goring-on-Thames.<br>ttft_avg: 0.068<br><br><br>             | output_text: Goring-on-Thames, Oxfordshire<br>ttft_avg: 0.068<br><br><br>       | output_text: Goring-on-Thames.<br>ttft_avg: 0.068<br><br><br>                | output_text: In Goring-on-Thames.<br>ttft_avg: 0.030<br><br><br>                    |
| output_sample_8  | Do director of film Betrayal (1932 Film) and director of film The Godsend (Film) share the same nationality? | output_text: Yes, both are French.<br>ttft_avg: 0.262<br><br>           | output_text: Yes, both are French directors.<br>ttft_avg: 0.074<br><br><br>  | output_text: Yes, both are French.<br>ttft_avg: 0.159<br><br><br>            | output_text: Both French.<br>ttft_avg: 0.068<br><br><br>                          | output_text: Both French.<br>ttft_avg: 0.068<br><br><br>                       | output_text: Both directors are French.<br>ttft_avg: 0.068<br><br><br>        | output_text: Both directors are French.<br>ttft_avg: 0.068<br><br><br>       | output_text: Yes, both are French.<br>ttft_avg: 0.068<br><br><br>               | output_text: Yes, both are French.<br>ttft_avg: 0.068<br><br><br>            | output_text: Both directors are French.<br>ttft_avg: 0.030<br><br><br>              |
| output_sample_9  | Which film whose director was born first, The Abduction Club or Wooden Crosses?                              | output_text: Wooden Crosses (1932)<br>ttft_avg: 0.242<br><br>           | output_text: Wooden Crosses (1932)<br>ttft_avg: 0.071<br><br><br>            | output_text: Wooden Crosses (1932)<br>ttft_avg: 0.148<br><br><br>            | output_text: Wooden Crosses (1932)<br>ttft_avg: 0.066<br><br><br>                 | output_text: Wooden Crosses was first released.<br>ttft_avg: 0.066<br><br><br> | output_text: Wooden Crosses was made first.<br>ttft_avg: 0.066<br><br><br>    | output_text: Wooden Crosses (1960)<br>ttft_avg: 0.066<br><br><br>            | output_text: Fernando Ayala (The Abduction Club)<br>ttft_avg: 0.066<br><br><br> | output_text: Wooden Crosses (1932)<br>ttft_avg: 0.066<br><br><br>            | output_text: Wooden Crosses (older)<br>ttft_avg: 0.029<br><br><br>                  |
| output_sample_10 | Which film came out earlier, Above Rubies or The Magic Aster?                                                | output_text: Above Rubies came out earlier.<br>ttft_avg: 0.263<br><br>  | output_text: Above Rubies came out earlier.<br>ttft_avg: 0.074               | output_text: Above Rubies came out earlier.<br>ttft_avg: 0.160<br><br><br>   | output_text: Above Rubies came out earlier.<br>ttft_avg: 0.068<br><br><br>        | output_text: Above Rubies came out earlier.<br>ttft_avg: 0.068<br><br><br>     | output_text: Above Rubies came out earlier.<br>ttft_avg: 0.068<br><br><br>    | output_text: Above Rubies came out earlier.<br>ttft_avg: 0.068<br><br><br>   | output_text: Above Rubies came out earlier.<br>ttft_avg: 0.068<br><br><br>      | output_text: Above Rubies came out earlier.<br>ttft_avg: 0.068<br><br><br>   | output_text: Above Rubies came out earlier.<br>ttft_avg: 0.030<br><br><br>          |