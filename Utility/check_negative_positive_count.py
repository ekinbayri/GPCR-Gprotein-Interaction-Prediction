import random

# Set the input and output file paths
#file_path = "GPCR_G_protein_labels_new.tsv"
file_path = "pan_pairs.tsv"

count_0 = 0
count_1 = 0

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:  # skip empty lines
            continue
        last_value = line.split("\t")[-1]
        if last_value == "0":
            count_0 += 1
        elif last_value == "1":
            count_1 += 1

print(f"Number of 0's: {count_0}")
print(f"Number of 1's: {count_1}")