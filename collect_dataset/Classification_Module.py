import os
import sys
import pdfplumber
import json
import shutil
from tqdm import tqdm


# 1. Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 2. Import API
try:
    from LLMs_API import client, get_model_name
    MODEL_NAME = get_model_name("pro") 
except ImportError:
    print("Error: Could not find LLMs_API.py in the parent directory.")
    sys.exit(1)

# ========= Functional Logic =========

def get_pdf_intro(pdf_path):
    """Extract the content of the first 2 pages of the PDF for classification."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i in range(min(2, len(pdf.pages))):
                text += pdf.pages[i].extract_text() or ""
    except Exception as e:
        print(f"Failed to parse {pdf_path}: {e}")
    return text[:3000]

def classify_paper(content):
    """Call Gemini for multi-dimensional classification determination."""
    system_prompt = """
    You are a senior research assistant in the field of optoelectronic materials. 
    Please determine the category of the provided paper content based on the following rules.
    
    【Classification Rules】：
    1. D-A TADF: Experimental literature on Thermally Activated Delayed Fluorescence (TADF), limited to small molecule structures.
    2. Theoretical: Literature focused purely on theoretical calculations or DFT simulations.
    3. Polymer: TADF literature involving polymers, macromolecules, or dendrimers.
    4. Metal: Literature on TADF complexes with metal centers (e.g., Ir, Pt, Cu, Au, Ce, etc.).
    5. MR-TADF: Experimental literature on Multi-Resonance TADF.
    6. Host materials: Experimental literature reporting the synthesis of host material molecules.
    7. Review: Review literature on TADF.
    8. Others: Literature that does not belong to the above categories or is not TADF-related.
    
    Please return ONLY the following JSON format:
    {
      "is_target": boolean,
      "reason": "Short reason for determination",
      "category": "D-A TADF / Theoretical / Polymer / Metal / MR-TADF / Host materials / Review / Others"
    }
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please classify this literature:\n\n{content}"}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"is_target": False, "reason": f"API Error: {e}", "category": "Error"}


def process_directory(source_folder, output_base_folder, global_pbar=None):
    """
    Process a single directory.
    Args:
        global_pbar: The shared tqdm progress bar object.
    """
    if not os.path.exists(source_folder):
        tqdm.write(f"Folder not found: {source_folder}")
        return

    files = [f for f in os.listdir(source_folder) if f.lower().endswith('.pdf')]
    # If using global bar, we don't print "Skipping empty" to avoid clutter, just return
    if not files:
        return

    # tqdm.write allows printing without breaking the progress bar layout
    tqdm.write(f"--- Entering folder: {os.path.basename(source_folder)} ---")

    for file in files:
        file_path = os.path.join(source_folder, file)
        
        # 1. Read PDF
        intro_text = get_pdf_intro(file_path)
        
        if len(intro_text) < 50:
            tqdm.write(f"[Skip] Text too short: {file}")
            # Even if skipped, we must update the progress bar because we "processed" it
            if global_pbar: global_pbar.update(1)
            continue
            
        # 2. AI Classification
        decision = classify_paper(intro_text)
        category = decision.get("category", "Others").strip()
        
        # 3. Path Handling
        safe_category = "".join([c for c in category if c.isalnum() or c in (' ', '-', '_')]).strip()
        
        category_folder = os.path.join(output_base_folder, safe_category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)
            
        target_path = os.path.join(category_folder, file)
        
        try:
            shutil.copy2(file_path, target_path)
            #mark = "[✓]" if decision.get("is_target") else "[ ]"
            #tqdm.write(f"{mark} {safe_category[:15].ljust(15)} : {file}")
        except Exception as e:
            tqdm.write(f"[!] Copy Failed: {file} - {e}")
        
        # 4. Update the global progress bar
        if global_pbar:
            global_pbar.update(1)


