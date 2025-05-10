import os
import json
from collections import defaultdict, Counter
from statistics import mean
import pandas as pd
import pprint

folders = [
    # "/u/zihengc2/multistage-sd/CacheBlend/select_full_recompute",
    "/u/zihengc2/multistage-sd/CacheBlend/select_imp_2nd_token",
    "/u/zihengc2/multistage-sd/CacheBlend/select_imp_3rd_layer",
    "/u/zihengc2/multistage-sd/CacheBlend/select_imp_4_20_layer",
    "/u/zihengc2/multistage-sd/CacheBlend/select_imp_8_24_layer",
    "/u/zihengc2/multistage-sd/CacheBlend/select_imp_12_28_layer",
    "/u/zihengc2/multistage-sd/CacheBlend/select_imp_16_32_layer",
    "/u/zihengc2/multistage-sd/CacheBlend/select_imp_from_k",
    "/u/zihengc2/multistage-sd/CacheBlend/select_imp_from_v_baseline",
    "/u/zihengc2/multistage-sd/CacheBlend/select_imp_no_selection"
]

results = {}

for folder in folders:
    folder_name = os.path.basename(folder)
    results[folder_name] = {}

    # Collect files
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    samples = defaultdict(list)

    # Group files by sample
    for f in files:
        parts = f.split("_")
        sample_id = f"{parts[2]}"
        samples[sample_id].append(os.path.join(folder, f))

    # Process each sample
    for sample_id, file_paths in samples.items():
        sample_data = {
            "output_texts": [],
            "output_files": [],
            "imp_indices": [],
            "ttft_with_cache": []
        }

        # Read all runs
        for file_path in sorted(file_paths):
            with open(file_path) as f:
                data = json.load(f)
                sample_data["output_texts"].append(data["output_text"])
                sample_data["output_files"].append(file_path)
                sample_data["imp_indices"].append(tuple(data["imp_indices"]))
                sample_data["ttft_with_cache"].append(data["ttft_with_cache"])

        def check_consistency(values):
            """Check if all values are the same."""
            return len(set(values)) == 1

        def get_inconsistent_files(values, files):
            """Return file paths where values differ from the most common."""
            counter = Counter(values)
            most_common_val, _ = counter.most_common(1)[0]
            inconsistent = [files[i] for i, v in enumerate(values) if v != most_common_val]
            return inconsistent

        # Check consistency
        output_text_consistent = check_consistency(sample_data["output_texts"])
        imp_indices_consistent = check_consistency(sample_data["imp_indices"])

        result_entry = {
            "output_consistent": output_text_consistent,
            "output_inconsistent_files": get_inconsistent_files(sample_data["output_texts"], sample_data["output_files"]),
            "imp_indices_consistent": imp_indices_consistent,
            "imp_indices_inconsistent_files": get_inconsistent_files(sample_data["imp_indices"], sample_data["output_files"]),
            "imp_indices": list(Counter(sample_data["imp_indices"]).most_common(1)[0][0]), 
            "ttft_avg": mean(sample_data["ttft_with_cache"]),
            "output_text": Counter(sample_data["output_texts"]).most_common(1)[0][0]
        }

        results[folder_name][f"output_sample_{sample_id}"] = result_entry

# pprint.pprint(results)

# Assuming `results` is already available from the previous step
folders = [
    # "select_full_recompute",
    "select_imp_2nd_token",
    "select_imp_3rd_layer",
    "select_imp_4_20_layer",
    "select_imp_8_24_layer",
    "select_imp_12_28_layer",
    "select_imp_16_32_layer",
    "select_imp_from_k",
    "select_imp_from_v_baseline",
    "select_imp_no_selection"
]

samples = [f"output_sample_{i}" for i in range(1, 11)]

table_data = []

for sample_id in samples:
    row = {}
    row["Sample"] = sample_id

    for folder in folders:
        if sample_id in results[folder]:
            entry = results[folder][sample_id]
            summary = (
                f'output_text: {entry["output_text"]}\n'
                f'ttft_avg: {entry["ttft_avg"]:.3f}\n'
                f'output_consistent: {entry["output_consistent"]}\n'
                f'imp_indices_consistent: {entry["imp_indices_consistent"]}\n'
            )
        else:
            summary = "N/A"

        row[folder] = summary

    table_data.append(row)

df = pd.DataFrame(table_data)

# Export to CSV
output_csv = "/u/zihengc2/multistage-sd/CacheBlend/example/summary_table.csv"
df.to_csv(output_csv, index=False)

# Print table
pd.set_option('display.max_colwidth', None)
print(df.to_string(index=False))
