import pdfplumber
import json
import csv
import re
import os
import sys
import time
from openai import OpenAI
from tqdm import tqdm

# 1. Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 2. Import API
from LLMs_API import client, get_model_name

print("Connecting to LLM model...") 

MODEL_NAME = get_model_name("pro") 
print("Model connected.")
    
    
# ========= Folder Paths =========
PUBLISHERS = ["acs", "elsevier", "nature", "rsc", "wiley"]
SUB_TYPES = ["D-A TADF", "MR-TADF"]
SI_DIR_NAME = "SI"
BASE_DIR_NAME = "Literatrue"

CSV_HEADERS = [
    "DOI", "TADF Name", "Solvent/Host", "Photosensitizer Name",
    "absorption_wavelength_nm", "emission_wavelength_nm", "FWHM_nm",
    "Delta_EST_eV", "PLQY_percent", "EQE_max_percent",
    "lifetime_us", "kRISC_x1e5_s-1", "kISC_x1e7_s-1",
    "kd_x1e5_s-1", "kr_x1e7_s-1", "knr_x1e7_s-1",
    "quotes", "notes"
]

def normalize_doi_for_matching(filename):
    """Normalize DOI by replacing the first separator with a hyphen """
    name_no_ext = os.path.splitext(filename)[0]
    # Standardize the first separator for matching 
    return re.sub(r"(——|—|_|-)", "-", name_no_ext.lower(), count=1)

def _as_numbers(x):
    """Format numerical values for CSV output [cite: 139, 140]"""
    if x is None: return ""
    if isinstance(x, (int, float)): return str(x)
    if isinstance(x, list):
        return " / ".join([str(i) for i in x if i is not None])
    return str(x)

def extract_pdf_to_snippets(pdf_path, label="MAIN"):
    """Parse PDF content and label snippets by source [cite: 11, 298]"""
    if not os.path.exists(pdf_path): return ""
    snippets = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Table Extraction [cite: 298]
                tables = page.extract_tables()
                for i, table in enumerate(tables):
                    if table:
                        table_str = "\n".join([" | ".join([str(cell) if cell else "" for cell in row]) for row in table])
                        snippets.append(f"[{label}][PAGE {page_num}][TABLE {i+1}]\n{table_str}")
                # Text Extraction [cite: 10, 11]
                text = page.extract_text()
                if text:
                    snippets.append(f"[{label}][PAGE {page_num}][TEXT]\n{text}")
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return "\n\n".join(snippets)

def append_to_csv(json_str, csv_path, doi_filename):
    """Append extracted JSON records to the master CSV"""
    try:
        data = json.loads(json_str)
        records = data.get("records", [])
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            if not file_exists:
                writer.writeheader()
            
            for rec in records:
                all_quotes = []
                sources = rec.get("sources", {})
                if isinstance(sources, dict):
                    for field, src_list in sources.items():
                        if isinstance(src_list, list):
                            for s in src_list:
                                quote = s.get("quote", "").strip()
                                if quote: all_quotes.append(quote)
                
                row = {
                    "DOI": doi_filename,
                    "TADF Name": str(rec.get("TADF Name") or "").strip(),
                    "Solvent/Host": str(rec.get("Solvent/Host") or "").strip(),
                    "Photosensitizer Name": str(rec.get("Photosensitizer Name") or "").strip(),
                    "absorption_wavelength_nm": _as_numbers(rec.get("absorption_wavelength_nm")),
                    "emission_wavelength_nm": _as_numbers(rec.get("emission_wavelength_nm")),
                    "FWHM_nm": _as_numbers(rec.get("FWHM_nm")),
                    "Delta_EST_eV": _as_numbers(rec.get("Delta_EST_eV")),
                    "PLQY_percent": _as_numbers(rec.get("PLQY_percent")),
                    "EQE_max_percent": _as_numbers(rec.get("EQE_max_percent")),
                    "lifetime_us": _as_numbers(rec.get("lifetime_us")),
                    "kRISC_x1e5_s-1": _as_numbers(rec.get("kRISC_x1e5_s-1")),
                    "kISC_x1e7_s-1": _as_numbers(rec.get("kISC_x1e7_s-1")),
                    "kd_x1e5_s-1": _as_numbers(rec.get("kd_x1e5_s-1")),
                    "kr_x1e7_s-1": _as_numbers(rec.get("kr_x1e7_s-1")),
                    "knr_x1e7_s-1": _as_numbers(rec.get("knr_x1e7_s-1")),
                    "quotes": " || ".join(list(set(all_quotes))),
                    "notes": str(rec.get("notes") or "").strip()
                }
                writer.writerow(row)
    except Exception as e:
        print(f"Failed to append to CSV: {e}")

def process_single_doi(main_pdf_path, si_pdf_path):
    """Call LLM API to extract and merge data from Paper and SI [cite: 345]"""
    #print(f"Processing: {os.path.basename(main_pdf_path)}")
    main_content = extract_pdf_to_snippets(main_pdf_path, label="PAPER_TABLES")
    si_content = extract_pdf_to_snippets(si_pdf_path, label="SI_TABLES")
    
    full_combined_content = main_content + "\n\n" + si_content
    # Limit content length to avoid token issues 
    #snippet_input = full_combined_content[:60000] 
    snippet_input = full_combined_content
    user_prompt = f"""
Extract ONLY EXPERIMENTAL photophysics/device properties for ALL TADF emitters from the SNIPPETS.
The snippets contain data from both the Main Paper and the Supporting Information (SI).

HARD filters:
- If the local context contains calculated / DFT / TD-DFT / computed / theoretical / simulated → IGNORE those numbers.
- Only create a record when BOTH "TADF Name" AND a concrete "Solvent/Host" are explicitly present in the SAME context.
- "Solvent/Host" MUST NOT be "device".
- During extraction, the host molecule percentage is not required.
- Data with the same host-guest names should be merged.
- If a photosensitizer is present, its name should also be included, such as photosensitizer name.
- A table may contain TADF properties under multiple HOSTs, and these should be distinguished.
- The delay fluorescence is usually in the range of microseconds, not instantaneous fluorescence time.
- Retain film properties first, then consider device properties for properties with the same subject and object names.

Extraction priority 1→2→3:
1) PAPER_TABLES
2) main TEXT
3) SI_TABLES

Dedup/merge:
- Carefully merge data for the same (TADF Name, Solvent/Host) if it is distributed across both the Main Paper and SI snippets.

Return ONE JSON object with exactly one key "records":
{{
  "records": [
    {{
      "TADF Name": "string",
      "Solvent/Host": "string",
      "Photosensitizer Name": "string",
      "absorption_wavelength_nm": number or [numbers] or null,
      "emission_wavelength_nm": number or [numbers] or null,
      "FWHM_nm": number or [numbers] or null,
      "Delta_EST_eV": number or [numbers] or null,
      "PLQY_percent": number or [numbers] or null,
      "EQE_max_percent": number or [numbers] or null,
      "lifetime_us": number or [numbers] or null,
      "kRISC_x1e5_s-1": number or [numbers] or null,
      "kISC_x1e7_s-1": number or [numbers] or null,
      "kd_x1e5_s-1": number or [numbers] or null,
      "kr_x1e7_s-1": number or [numbers] or null,
      "knr_x1e7_s-1": number or [numbers] or null,
      "notes": "string",
      "sources": {{
        "any_field": [{{"page": int, "type": "text|table", "quote": "string"}}]
      }}
    }}
  ]
}}

- Output ONLY the JSON object.

SNIPPETS:
{full_combined_content}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a specialized TADF data extraction assistant."},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error for {os.path.basename(main_pdf_path)}: {e}")
        return None

def normalize_for_comparison(filename):
    """
    强力匹配辅助函数：
    1. 去掉扩展名
    2. 去掉末尾的 -si 或 _si (忽略大小写)
    3. 将所有点(.)、下划线(_)、中文横线(——)都替换为标准短横线(-)
    4. 全部转为小写
    这样 '10.1038/lsa.2015.5' 无论中间怎么变都能对上
    """
    name = os.path.splitext(filename)[0].lower()
    # 去掉 si 后缀
    name = re.sub(r'[-_]si$', '', name)
    # 将所有干扰分隔符统一
    normalized = re.sub(r'[.———_]', '-', name)
    return normalized

def count_total_files(base_dir):
    """预扫描所有文件夹，计算总任务数"""
    total = 0
    print("Scanning files for total count...")
    for publisher in PUBLISHERS:
        pub_root = os.path.join(base_dir, publisher)
        for sub_type in SUB_TYPES:
            target_folder = os.path.join(pub_root, sub_type)
            if os.path.exists(target_folder):
                files = [f for f in os.listdir(target_folder) if f.lower().endswith('.pdf')]
                total += len(files)
    print(f"Total files to process: {total}")
    return total

def process_folder_logic(main_folder, si_folder, output_csv, pbar):
    """
    处理单个文件夹
    """
    if not os.path.exists(main_folder):
        return

    # 1. 建立 SI 查找表
    si_map = {}
    if os.path.exists(si_folder):
        for f in os.listdir(si_folder):
            if f.lower().endswith(('.pdf', '.docx', '.doc')):
                fingerprint = normalize_for_comparison(f)
                si_map[fingerprint] = f

    # 2. 获取主文献列表
    main_files = [f for f in os.listdir(main_folder) if f.lower().endswith('.pdf')]

    # 3. 循环处理
    for main_file in main_files:
        # [核心修改] 这里不再调用 pbar.set_description 显示文件名
        # 保持静默，只更新进度条的长度
        
        main_path = os.path.join(main_folder, main_file)
        
        # 匹配 SI
        main_fingerprint = normalize_for_comparison(main_file)
        si_file = si_map.get(main_fingerprint)
        si_path = os.path.join(si_folder, si_file) if si_file else ""
        
        if not si_path:
            extreme_fingerprint = re.sub(r'[^a-z0-9]', '', main_fingerprint)
            for k, v in si_map.items():
                if re.sub(r'[^a-z0-9]', '', k) == extreme_fingerprint:
                    si_path = os.path.join(si_folder, v)
                    break
        
        # 调用 API
        result_json = process_single_doi(main_path, si_path)
        if result_json:
            append_to_csv(result_json, output_csv, main_file)
        
        # 更新总进度条 (每次完成一个文件 +1)
        pbar.update(1)

def main():
    base_dir = BASE_DIR_NAME 

    # 1. 计算总数
    total_files = count_total_files(base_dir)
    if total_files == 0:
        print("No files found.")
        return

    print("=== Starting to extract data ===")

    # 2. 初始化总进度条
    # ncols控制进度条宽度，避免太长换行
    with tqdm(total=total_files, unit="file", ncols=100) as pbar:
        
        # 外层循环：遍历5个出版社
        for publisher in PUBLISHERS:
            # [核心修改] 在这里更新进度条的描述，只显示出版社名字
            pbar.set_description(f"Processing Publisher: {publisher}")
            
            pub_root = os.path.join(base_dir, publisher)
            current_si_folder = os.path.join(pub_root, SI_DIR_NAME)
            final_csv_path = f"{publisher}_TADF_All_Results.csv"
            
            # 内层循环：遍历 D-A TADF 和 MR-TADF
            for sub_type in SUB_TYPES:
                current_main_folder = os.path.join(pub_root, sub_type)
                process_folder_logic(current_main_folder, current_si_folder, final_csv_path, pbar)

    print(f"\n=== All publishers processed. Data saved ===")

if __name__ == "__main__":
    main()
