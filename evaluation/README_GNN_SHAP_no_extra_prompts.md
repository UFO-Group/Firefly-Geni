# GNN-SHAP evaluation scripts with fewer prompts

Place these files in `Firefly-Geni/evaluation/`.

This version keeps Step 1 environment selection, and reduces Step 2/Step 3 prompts.

## Step 2 asks only:
- target property
- MC_STEPS
- CUDA device id

The following are fixed automatically:
- PRE_DIR from Step 1 config or `../dualmol-net`
- model weights from `save_end-to-end_10fold/fold_5/best_avg_r2_model.pth`
- load_whole_model = True
- random seed = 42
- mask baseline = zero
- compute TADF atom SHAP = True
- compute host atom SHAP = True
- require Step 1 match = True

## Step 3 asks only:
- target property
- MC_STEPS
- TADF fragment atom indices
- CUDA device id

The following are fixed automatically:
- PRE_DIR
- model weights
- load_whole_model = True
- random seed = 42
- mask baseline = zero
- compute TADF fragment SHAP = True
- compute host fragment SHAP = False
- add other_atoms fragment = True
- require Step 1 match = True

Branch-level SHAP is removed.
