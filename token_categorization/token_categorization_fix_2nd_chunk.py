import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# Pattern of files
base_dir = "/u/zihengc2/multistage-sd/CacheBlend/output_split/"
patterns = ["1_0_1", "1_2_1", "1_3_1", "1_4_1", "1_5_1"]
all_imp_indices = []

# Loop through each pattern
for pattern in patterns:
    file_path = os.path.join(base_dir, f"{pattern}/output_sample_{pattern}_run_1.json")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist, skipping.")
        continue
    
    # Load JSON and extract imp_indices
    with open(file_path, "r") as f:
        data = json.load(f)
        imp_indices = data.get("imp_indices", [])
        
        # Remove last 61 elements
        if len(imp_indices) > 61:
            imp_indices = imp_indices[:-61]
        else:
            imp_indices = []
        
        # Filter out numbers smaller than 20
        imp_indices = [x for x in imp_indices if x >= 20]
        imp_indices = [x - imp_indices[0] for x in imp_indices]
        
        all_imp_indices.append(imp_indices)

index_count = np.zeros(625, dtype=int)

for indices in all_imp_indices:
    for idx in indices:
        index_count[idx] += 1

total_lists = len(all_imp_indices)

# Categorize indices
always_picked = np.where(index_count == total_lists)[0]
never_picked = np.where(index_count == 0)[0]
sometimes_picked = np.where((index_count > 0) & (index_count < total_lists))[0]

# Plot
plt.figure(figsize=(8, 3), dpi=200)
plt.scatter(never_picked, [1] * len(never_picked), color='gray', label='Dummy (Never Picked)', marker='|', s=500)
plt.scatter(always_picked, [1] * len(always_picked), color='purple', label='Relational (Always Picked)', marker='|', s=500)
plt.scatter(sometimes_picked, [1] * len(sometimes_picked), color='red', label='Sometimes Picked', marker='|', s=1000)

plt.xlabel("Indices")
plt.yticks([])
plt.title("Global Token Categorization")
plt.legend()
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig(f"token_categorization_chunk_1.png")