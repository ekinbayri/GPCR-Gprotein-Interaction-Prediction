import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class GPCRDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        gpcr = row["GPCR"]
        g_protein = row["G-protein"]
        label = torch.tensor(row["label"], dtype=torch.float32)
        return gpcr, g_protein, label


csv_file = "GPCR_G_protein_labels.csv" 
dataset = GPCRDataset(csv_file)

batch64_loader = DataLoader(dataset, batch_size=64, shuffle=True)
batch32_loader = DataLoader(dataset, batch_size=32, shuffle=True)
batch16_loader = DataLoader(dataset, batch_size=16, shuffle=True)


print("Batch Size 64:")
for gpcrs, g_proteins, labels in batch64_loader:
    print("GPCRs:", gpcrs)
    print("G-proteins:", g_proteins)
    print("Labels:", labels)
    break  

print("\nBatch Size 32:")
for gpcrs, g_proteins, labels in batch32_loader:
    print("GPCRs:", gpcrs)
    print("G-proteins:", g_proteins)
    print("Labels:", labels)
    break

print("\nBatch Size 16:")
for gpcrs, g_proteins, labels in batch16_loader:
    print("GPCRs:", gpcrs)
    print("G-proteins:", g_proteins)
    print("Labels:", labels)
    break
