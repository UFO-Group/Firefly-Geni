# Dataset Sample Checks

Three samples of 100 records each were randomly drawn from `../dataset_tadf/all_data_with_smiles.csv` using fixed seeds of 42, 64, and 128. These samples were used to manually assess SMILES accuracy, extracted-data accuracy, and the proportion of experimental versus theoretical data. The samples were drawn independently and may overlap.

- **CSV files:** Original sampled records before manual annotation.
- **XLSX files:** Manually checked records. Red cell fills indicate incorrect entries; green cell fills indicate theoretical data.

All three samples contained 100% experimental data (experimental:theoretical = 100:0), so no green fills were applied. Red fills, where present, mark errors identified during manual checking.