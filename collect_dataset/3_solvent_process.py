import pandas as pd
import re
import os
import time
# =============================================================================
# PART 1: Extract Host Name
# Functionality: Removes doping concentrations (e.g., "10 wt%", "(5 wt%)") 
# from the 'Solvent/Host' string to isolate the host material name.
# =============================================================================

def extract_host_name(s: str) -> str:
    """
    Extracts host substrate name from strings like 'ppf (10 wt%)' / 'cbp (3 wt%)' / '1 wt% pmma'.
    Rules:
    - Remove doping/weight fraction info in parentheses.
    - Support prefix ratios: '1 wt% pmma' -> 'pmma'.
    - Output normalized to lowercase, stripped of whitespace.
    """
    if pd.isna(s):
        return ""
    
    # Normalize to string, strip whitespace, convert to lowercase
    s = str(s).strip().lower()
    if s == "" or s == "nan":
        return ""

    # 1) Handle prefix ratios first: "10 wt% doped pmma" / "1 wt% pmma" etc.
    s = re.sub(
        r'^\s*\d+(?:\.\d+)?\s*(?:wt|mol|vol)\s*%?\s*(?:doped|doping|processed|in|of|with)?\s*',
        '',
        s
    ).strip()

    # 2) Remove content in parentheses (e.g., "(10 wt%)", "(1.0 wt%)")
    s = re.sub(r'\s*\([^)]*\)\s*', ' ', s).strip()

    # 3) Remove remaining isolated ratio fragments (rare cases)
    s = re.sub(r'\b\d+(?:\.\d+)?\s*(?:wt|mol|vol)\s*%?\b', '', s).strip()

    # 4) Merge excess spaces and strip potential residual quotes
    s = re.sub(r'\s+', ' ', s).strip().strip('"').strip("'")

    return s

def run_extraction_step(in_file, out_file, col):
    print("\n--- [Step 1] Extracting Host Names ---")
    
    if not os.path.exists(in_file):
        print(f"Error: Input file '{in_file}' not found.")
        return

    df = pd.read_csv(in_file, encoding="utf-8-sig")

    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found. Available columns: {df.columns.tolist()}")

    # Apply extraction logic directly to the column
    df[col] = df[col].apply(extract_host_name)

    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"✅ Step 1 Complete! Modified column '{col}' saved to: {out_file}")


# =============================================================================
# PART 2: Standardize "Pure Solid"
# Functionality: Replaces various terms indicating a neat film, powder, or 
# crystal (e.g., "neat film", "powder") with the standardized term "pure solid".
# =============================================================================

def replace_with_pure_solid(in_file, out_file, col):
    print("\n--- [Step 2] Standardizing 'Pure Solid' Terms ---")
    
    if not os.path.exists(in_file):
        print(f"Error: Input file '{in_file}' not found.")
        return

    df = pd.read_csv(in_file, encoding="utf-8-sig")
    
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")

    # List of keywords to map to "pure solid"
    pure_solid_keywords = [
        "neat film", "neat", "powder", "solid state", "thin film", "solid powder", "crystal",
        "film", "solid", "pure film", "crystalline powder", "single crystal", 
        "amorphous powder", "ground solid", "ground powder", "polycrystalline powder", 
        "amorphous", "ground sample", "neat film (pristine)", "pristine film", 
        "crystalline", "pristine solid", "neat thin films", "neat film (drop-casted)", 
        "neat film (ground)", "y-crystal", "crystalline state", "non-doped", 
        "non-doped film", "r-crystal", "neat film (amorphous)", "microcrystal", 
        "neat film (crystalline)", "crystal a", "crystal b", "powder (ground)", 
        "powder (initial)", "neat solid", "yellow solid (ys)", "crystal c2", 
        "crystals", "compressed film", "fumed solid", "crystal 1", "crystal 2", 
        "or crystal", "gy crystal", "h_r-solid", "crystal-y", "crystal-r", 
        "f_y-solid", "o-crystal", "y-solid", "single crystals", "crystal c3", 
        "crystal c4", "crystal c5", "orange solid", "orange crystal", 
        "red crystal", "solids", "nondoped", "as-prepared solid", 
        "microcrystalline powder", "macrocrystals", "crystal with h2o", 
        "crystal without h2o", "crystal ctm-b", "crystal ctm-w", 
        "crystal tctm-b", "crystal tctm-w", "microcrystals", "g-crystal", 
        "heated powder", "crystalline solid", "amorphous solid", "annealed film"
    ]

    # Normalize column to string, strip whitespace, and convert to lowercase
    df[col] = df[col].astype(str).str.strip().str.lower()
    
    # Bulk replacement: replace any value in the keyword list with "pure solid"
    df[col] = df[col].replace(pure_solid_keywords, "pure solid")

    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    
    count = (df[col] == 'pure solid').sum()
    print(f"✅ Step 2 Complete! Standardized keywords to 'pure solid'.")
    print(f"📊 Stats: Count of 'pure solid' in '{col}': {count}")


# =============================================================================
# PART 3: Normalize Solvent/Host Names
# Functionality: Uses a comprehensive dictionary to map various abbreviations
# (e.g., "dcm", "ch2cl2") to standardized chemical names (e.g., "dichloromethane").
# =============================================================================

# Comprehensive Dictionary for Normalization
SOLVENT_HOST_MAP = {
    # --- DCM ---
    "dcm": "dichloromethane",
    "ch2cl2": "dichloromethane",
    "methylene chloride": "dichloromethane",
    "dichloromethane": "dichloromethane",

    # --- Chloroform ---
    "chcl3": "chloroform",
    "trichloromethane": "chloroform",
    "chloroform": "chloroform",

    # --- THF ---
    "tetrahydrofuran": "thf",
    "thf": "thf",
    "c4h8o": "thf",

    # --- 2-MeTHF ---
    "2-methf": "2-methyltetrahydrofuran",
    "2methf": "2-methyltetrahydrofuran",
    "methf": "2-methyltetrahydrofuran",
    "mthf": "2-methyltetrahydrofuran",
    "2-me-thf": "2-methyltetrahydrofuran",
    "2me-thf": "2-methyltetrahydrofuran",
    "me-thf": "2-methyltetrahydrofuran",
    "methyl-thf": "2-methyltetrahydrofuran",
    "2-methyl-tetrahydrofuran": "2-methyltetrahydrofuran",
    "2-methyl tetrahydrofuran": "2-methyltetrahydrofuran",
    "2-methyltetrahydrofuran": "2-methyltetrahydrofuran",

    # --- Acetonitrile ---
    "mecn": "acetonitrile",
    "ch3cn": "acetonitrile",
    "acn": "acetonitrile",
    "acetonitrile": "acetonitrile",

    # --- DMF ---
    "dimethylformamide": "dmf",
    "n,n-dimethylformamide": "dmf",
    "dimethyl formamide": "dmf",
    "dimethylfomamide": "dmf",
    "dmf": "dmf",

    # --- DMSO ---
    "dimethyl sulfoxide": "dmso",
    "dimethylsulfoxide": "dmso",
    "dimethylsulphoxide": "dmso",
    "dmso": "dmso",

    # --- Alcohols ---
    "meoh": "methanol",
    "methyl alcohol": "methanol",
    "methanol": "methanol",
    "etoh": "ethanol",
    "ethyl alcohol": "ethanol",
    "ethanol": "ethanol",
    "isopropanol": "isopropanol",
    "ipa": "isopropanol",
    "propanol": "propanol",
    "n-propanol": "propanol",
    "butanol": "butanol",
    "n-butanol": "butanol",
    "1-butanol": "butanol",

    # --- Ethers ---
    "diethylether": "diethyl ether",
    "ethyl ether": "diethyl ether",
    "et2o": "diethyl ether",
    "diethyl ether": "diethyl ether",
    "ether": "diethyl ether",
    "isopropylether": "isopropyl ether",
    "isopropyl ether": "isopropyl ether",
    "butylether": "butyl ether",
    "butyl ether": "butyl ether",
    "dibutylether": "dibutyl ether",
    "dibutyl ether": "dibutyl ether",
    "bu2o": "dibutyl ether",
    "1,4-dioxane": "dioxane",
    "dioxane": "dioxane",

    # --- Hydrocarbons ---
    "tol": "toluene",
    "phme": "toluene",
    "methylbenzene": "toluene",
    "toluene": "toluene",
    "benzene": "benzene",
    "bz": "benzene",
    "hex": "hexane",
    "n-hexane": "hexane",
    "c6h14": "hexane",
    "hexane": "hexane",
    "n-heptane": "heptane",
    "heptane": "heptane",
    "c6h12": "cyclohexane",
    "chx": "cyclohexane",
    "cyh": "cyclohexane",
    "cyclohexane": "cyclohexane",
    "mch": "methylcyclohexane",
    "methylcyclohexane": "methylcyclohexane",
    "chlorobenzene": "chlorobenzene",
    "c6h5cl": "chlorobenzene",
    "mcb": "chlorobenzene",
    "mono chloro benzene": "chlorobenzene",
    "mesitylene": "mesitylene",
    "1,3,5-trimethylbenzene": "mesitylene",
    "anisole": "anisole",
    "o-dcb": "o-dichlorobenzene",
    "o-dichlorobenzene": "o-dichlorobenzene",
    "1,2-dichlorobenzene": "o-dichlorobenzene",
    "dcb": "dichlorobenzene",
    "dichlorobenzene": "dichlorobenzene",
    "o-xylene": "o-xylene",
    "o-xylenes": "o-xylene",
    "m-xylene": "m-xylene",
    "m-xylenes": "m-xylene",
    "p-xylene": "p-xylene",
    "p-xylenes": "p-xylene",
    "ccl4": "carbon tetrachloride",

    # --- Esters/Ketones ---
    "etoac": "ethyl acetate",
    "ea": "ethyl acetate",
    "eac": "ethyl acetate",
    "ethyl acetate": "ethyl acetate",
    "acetone": "acetone",
    "ace": "acetone",
    "ac": "acetone",
    "me2co": "acetone",
    "4-methoxybenzophenone": "4-methoxybenzophenone",

    # --- Water ---
    "deionized water": "water",
    "di water": "water",
    "h2o": "water",
    "water": "water",

    # --- Pyridine ---
    "pyridine": "pyridine",
    "py": "pyridine",

    # --- Polymers/Matrices ---
    "polymethylmethacrylate": "pmma",
    "pmma": "pmma",
    "polystyrene (ps)": "ps",
    "poly(styrene)": "ps",
    "polystyrene": "ps",
    "ps": "ps",
    "pvk": "pvk",
    "pva": "pva",
    "pvp": "pvp",
    "pvc": "pvc",
    "zeonex 480r": "zeonex",
    "zeonex 480": "zeonex",
    "zeonex2": "zeonex",
    "zeonex1": "zeonex",
    "znx": "zeonex",
    "zeonex": "zeonex",
    "polysulfone": "polysulfone",
    "pla": "pla",
    "poly(vinylidene chloride-co-acrylonitrile)": "poly(vinylidene chloride-co-acrylonitrile)",

    # --- OLED Hosts ---
    "cbp": "cbp",
    "m-cbp": "mcbp",
    "o-cbp": "o-cbp",
    "mcbp": "mcbp",
    "mcbp-cn": "mcbp-cn",
    "mcpcn": "mcpcn",
    "mcbpcn": "mcbp-cn",
    "mcp": "mcp",
    "m-cp": "mcp",
    "mcp-cn": "mcpcn",
    "mcpbc": "mcpbc",
    "mcpbp": "mcpbp",
    "mcppfp": "mcppfp",
    "mcp-pfp": "mcppfp",
    "ppf": "ppf",
    "ppt": "ppt",
    "dpepo": "dpepo",
    "bis[2-(diphenylphosphino)phenyl] ether oxide": "dpepo",
    "bis[2-(diphenylphosphino)phenyl]ether oxide": "dpepo",
    "tpbi": "tpbi",
    "tcta": "tcta",
    "czsi": "czsi",
    "cztrz": "cztrz",
    "bcpo": "bcpo",
    "dbfpo": "dbfpo",
    "pbict": "pbict",
    "pyd2": "pyd2",
    "simcp2": "simcp2",
    "26dczppy": "26dczppy",
    "2,6-dczppy": "26dczppy",
    "26-dczppy": "26dczppy",
    "3,5-dczppy": "35dczppy",
    "o-czoxd": "o-czoxd",
    "dpetpo": "dpetpo",
    "mcppy2po": "mcppy2po",
    "sicz": "sicz",
    "tmpypb": "tmpypb",
    "dmic-trz": "dmic-trz",
    "phcbbcz": "phcbbcz",
    "dbfdpo": "dbfdpo",
    "dic-trz": "dic-trz",
    "czacsf": "czacsf",
    "tcz1": "tcz1",
    "o-dicbzbz": "o-dicbzbz",
    "dobna-oar": "dobna-oar",
    "tspo1": "tspo1",
    "phenyl benzoate (phb)": "phenyl benzoate",
    "phenyl benzoate": "phenyl benzoate",
    "phb": "phenyl benzoate",
    "po-t2t": "po-t2t",
    "cdbp": "cdbp",
    "sf3trz": "sf3trz",
    "phczbcz": "phczbcz",
    "meobp": "meobp",
    "ttpo": "ttpo",
    "cz-3czcn": "cz-3czcn",
    "sitrzcz2": "sitrzcz2",
    "o-mcpbi": "o-mcpbi",
    "dmac-dps": "dmac-dps",
    "bepp2": "bepp2",
    "tapc": "tapc",
    "fppo": "fppo",
    "ugh": "ugh",
    "sppo1": "sppo1",
    "siczcz": "siczcz",
}

def clean_and_normalize(s):
    """Clean string and map to normalized value using dictionary."""
    if pd.isna(s):
        return ""
    
    # Thorough cleaning: remove quotes, strip whitespace, convert to lowercase
    clean_key = str(s).strip().strip('"').strip("'").lower()
    
    # Look up in map; if not found, return the cleaned key itself
    return SOLVENT_HOST_MAP.get(clean_key, clean_key)

def normalize_host_names(in_file, out_file, col):
    print("\n--- [Step 3] Normalizing Solvent/Host Names ---")
    
    if not os.path.exists(in_file):
        print(f"Error: Input file '{in_file}' not found.")
        return

    df = pd.read_csv(in_file, encoding="utf-8-sig")

    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")

    # Apply normalization using the dictionary
    df[col] = df[col].apply(clean_and_normalize)

    df.to_csv(out_file, index=False, encoding="utf-8-sig")

    print(f"✅ Step 3 Complete! Final file saved to: {out_file}")
    #print("-" * 30)
    #print("Top 10 Most Common Environments:")
    #print(df[col].value_counts().head(10))
    print("Solution information processed successfully")


# =============================================================================
# Main Execution Flow
# =============================================================================

if __name__ == "__main__":

    # Configuration
    target_column = "Solvent/Host"
    
    # File Paths for the sequential process
    file_step1_in = "all_data_with_smiles_process.csv"
    file_step1_out = "all_data_with_smiles_process_host_extracted.csv"
    
    file_step2_out = "all_data_with_smiles_process_host_extracted_pure_solid.csv"
    
    file_step3_out = "all_data_with_smiles_process_host_extracted_pure_solid_normalized.csv"

    # Step 1: Extract Host Name
    run_extraction_step(file_step1_in, file_step1_out, target_column)

    # Step 2: Standardize "Pure Solid"
    # Input is the output of Step 1
    if os.path.exists(file_step1_out):
        replace_with_pure_solid(file_step1_out, file_step2_out, target_column)
    
    # Step 3: Normalize Names
    # Input is the output of Step 2
    if os.path.exists(file_step2_out):
        normalize_host_names(file_step2_out, file_step3_out, target_column)
        
