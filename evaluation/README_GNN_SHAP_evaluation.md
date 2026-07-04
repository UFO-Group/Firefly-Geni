# Firefly-Geni evaluation GNN-SHAP scripts

Put these files in:

```text
Firefly-Geni/evaluation/
```

You also need the working package folder from your old runnable version:

```text
Firefly-Geni/evaluation/GNNshap_module/
```

Recommended folder structure:

```text
Firefly-Geni/evaluation/
├── GNN-SHAP-1.py
├── GNN-SHAP-2.py
├── GNN-SHAP-3.py
├── gnn_shap_runtime_inputs.py
└── GNNshap_module/
```

Run:

```bash
cd Firefly-Geni/evaluation
python GNN-SHAP-1.py
python GNN-SHAP-2.py
python GNN-SHAP-3.py
```

Main changes:
- Paths are relative to Firefly-Geni/evaluation.
- Step 1 prints host/environment list and accepts index, name, or direct SMILES.
- Step 2 and Step 3 automatically reuse the run config from Step 1.
- Output folders include molecule name and environment label.
- Branch-level SHAP imports, switches, calculations, and outputs are removed.
