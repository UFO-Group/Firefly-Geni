import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit import DataStructs

# Completely disable all RDKit C++-level error outputs
RDLogger.DisableLog('rdApp.*')


def get_canonical_valid(smiles_list):
    """
    Internal helper function:
    filter out invalid molecules and convert valid molecules into canonical SMILES.
    """
    valid_canonical = []

    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)

            if mol is not None:
                valid_canonical.append(Chem.MolToSmiles(mol, isomericSmiles=False))

        except:
            continue

    return valid_canonical


# ==========================================
# 1. Calculate SMILES validity
# ==========================================
def valid_molecules(molecules):
    """
    Calculate validity:
    number of valid molecules / total number of generated molecules.
    """
    valid_canonical = get_canonical_valid(molecules)

    valid_count = len(valid_canonical)
    valid_ratio = valid_count / len(molecules) if len(molecules) > 0 else 0.0

    return valid_count, valid_ratio


# ==========================================
# 2. Calculate uniqueness
# ==========================================
def uniqueness(molecules):
    """
    Calculate uniqueness:
    number of unique canonical valid molecules / total number of canonical valid molecules.

    This avoids artificially high uniqueness caused by invalid or corrupted SMILES strings.
    """
    valid_canonical = get_canonical_valid(molecules)

    if not valid_canonical:
        return 0.0

    unique_count = len(set(valid_canonical))

    return unique_count / len(valid_canonical)


# ==========================================
# 3. Calculate novelty
# ==========================================
def novelty(generated_molecules, training_set):
    """
    Calculate novelty:
    number of unique valid generated molecules not present in the training set
    / number of unique valid generated molecules.

    The comparison is based on canonicalized chemical structures to avoid
    superficial SMILES-format differences.
    """
    valid_gen = get_canonical_valid(generated_molecules)

    if not valid_gen:
        return 0.0

    unique_gen_set = set(valid_gen)

    # Make sure the training set is also canonicalized for a fair comparison
    train_canonical_set = set(get_canonical_valid(training_set))

    novel_mols = unique_gen_set - train_canonical_set

    return len(novel_mols) / len(unique_gen_set)


# ==========================================
# 4. Calculate diversity, stable CPU-only version
# ==========================================
def calculate_diversity(smiles_list):
    """
    Calculate diversity:
    1 - average Tanimoto similarity.

    The calculation is performed only on valid and deduplicated molecules.
    This CPU-only version uses RDKit's C++ backend to avoid GPU memory overflow.
    """
    # Extract valid and unique molecules
    unique_valid_canonical = list(set(get_canonical_valid(smiles_list)))

    if len(unique_valid_canonical) < 2:
        return 0.0

    # 1. Calculate Morgan fingerprints.
    # Note: the fingerprints are kept as RDKit bit-vector objects
    # instead of being converted into NumPy arrays.
    fps = []

    for smi in unique_valid_canonical:
        mol = Chem.MolFromSmiles(smi)

        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
            fps.append(fp)

    if len(fps) < 2:
        return 0.0

    # 2. Use RDKit's native BulkTanimotoSimilarity for efficient pairwise similarity calculation.
    # This method runs through the C++ backend and greatly reduces memory usage and CPU time.
    similarities = []
    n_fps = len(fps)

    for i in range(n_fps - 1):
        # Compare the fingerprint of molecule i with all following molecules,
        # from i + 1 to the end.
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        similarities.extend(sims)

    # 3. Calculate diversity
    if not similarities:
        return 0.0

    average_similarity = sum(similarities) / len(similarities)
    diversity = 1.0 - average_similarity

    return diversity


# ==========================================
# 5. Calculate token-level and sequence-level reconstruction accuracy
# ==========================================
def accu(pred, val, batch_l):
    correct = 0
    total = 0
    cor_seq = 0

    for i in range(0, batch_l.shape[0]):
        try:
            mm = (
                pred[i, 0:batch_l[i]].cpu().data.numpy()
                == val[i, 0:batch_l[i]].cpu().data.numpy()
            )

            correct += mm.sum()
            total += batch_l[i].sum()
            cor_seq += mm.all()

        except:
            return 0, 0

    acc = correct / float(total) if total > 0 else 0
    acc2 = cor_seq / batch_l.shape[0] if batch_l.shape[0] > 0 else 0

    return acc, acc2