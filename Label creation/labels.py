import pandas as pd


data = pd.read_csv('C:\\Users\Ekinb\\Downloads\\LogRAi_values_final.tsv', sep='\t')
data = data.rename(columns={"#Gene": "GPCR"})
output_data = []

for _, row in data.iterrows():
    gpcr = row['GPCR'] 
    for g_protein, affinity in row.items():
        if g_protein == 'GPCR':  
            continue
        label = 1 if affinity >= -1 else 0
        output_data.append({"GPCR": gpcr, "G-protein": g_protein, "label": label})


output_df = pd.DataFrame(output_data)
output_df.to_csv('GPCR_G_protein_labels.csv', index=False)

print("Transformation complete. Output saved to 'GPCR_G_protein_labels.csv'.")
