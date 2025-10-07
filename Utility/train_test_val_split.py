import random

# Set the input and output file paths
input_file = "GPCR_G_protein_labels.tsv"
store_file = "GPCR2_store.tsv"
train_file = "GPCR_train.tsv"
val_file = "GPCR_val.tsv"
test_file = "GPCR_test.tsv"

# Read and shuffle the lines
with open(input_file, "r") as f:
    lines = [line for line in f if line.strip()]

random.shuffle(lines)

# Split 80% for store, 20% for test
total = len(lines)
store_size = int(0.8 * total)
store_lines = lines[:store_size]
test_lines = lines[store_size:]

# Save test set
with open(test_file, "w") as f:
    f.writelines(test_lines)

# Split store into train (80%) and val (20%)
train_size = int(0.8 * store_size)
train_lines = store_lines[:train_size]
val_lines = store_lines[train_size:]

with open(train_file, "w") as f:
    f.writelines(train_lines)
with open(val_file, "w") as f:
    f.writelines(val_lines)
