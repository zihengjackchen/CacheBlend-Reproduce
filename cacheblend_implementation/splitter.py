import os
import json
from itertools import product

input_dir = "/u/zihengc2/multistage-sd/CacheBlend/inputs"
output_dir = "/u/zihengc2/multistage-sd/CacheBlend/inputs_split"

os.makedirs(output_dir, exist_ok=True)

# Process files 1.json to 10.json
for file_id in range(1, 11):
    input_path = os.path.join(input_dir, f"{file_id}.json")

    # Load the input file
    with open(input_path, "r") as f:
        data = json.load(f)

    # Extract chunk keys (assume they are strings of integers like "0", "1", "2", ...)
    chunk_keys = [key for key in data.keys() if key.isdigit()]
    query = data["query"]

    # Generate all ordered pairs (permutations with replacement)
    for i in chunk_keys:
        for j in chunk_keys:
            # Create the output dictionary
            output_data = {
                "0": data[i],
                "1": data[j],
                "query": query,
                "chunk_num": 2
            }

            # Write to output file
            output_filename = f"{file_id}_{i}_{j}.json"
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, "w") as out_f:
                json.dump(output_data, out_f, ensure_ascii=False, indent=2)

print("Done generating combinations.")
