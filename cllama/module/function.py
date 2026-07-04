import pandas as pd

def calculate_smiles_length(smiles, charset_double):
    """
    Calculate the actual token length of a single SMILES.
    Treat diatomic atoms (such as Si, Cl, Se, Br, Ge) as 1 unit length.
    """
    if pd.isna(smiles) or not isinstance(smiles, str):
        return 0
        
    length = 0
    i = 0
    while i < len(smiles):
        # Prioritize determining whether it is a two-letter atom
        if i + 1 < len(smiles) and smiles[i:i+2] in charset_double:
            length += 1
            i += 2
        else:
            length += 1
            i += 1
    return length