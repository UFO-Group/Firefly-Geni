import fitz  # PyMuPDF
import os
import json
import base64
import pandas as pd
import re
import shutil
import time
import sys
import glob
from PIL import Image
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
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
print(f"Model connected: {MODEL_NAME}")

MAX_WORKERS = 6  # Number of parallel threads

# ========= Configuration =========
PUBLISHERS = ["acs", "elsevier", "nature", "rsc", "wiley"]
SUB_TYPES = ["D-A TADF", "MR-TADF"]
SI_DIR_NAME = "SI"
BASE_DIR_NAME = "Literatrue" # 保持你的拼写
OUTPUT_FOLDER_NAME = "mol-picture-TADF" # 输出文件夹名

# Supplemental keywords for page filtering
CONTEXT_KEYWORDS = ["structure", "chromophore", "fluorescence", "synthesis", "experimental",
                    "structures", "chromophores", "fluorescences",]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_target_names_for_doi(df, doi):
    """Filter target names from the dataframe for a specific DOI and apply blacklist."""
    subset = df[df['DOI'].str.contains(doi, na=False)]
    columns = ["TADF Name", "Solvent/Host", "Photosensitizer Name"]
    all_names = []
    
    for col in columns:
        if col in subset.columns:
            # 提取非空值并转换为字符串
            vals = subset[col].dropna().unique().tolist()
            all_names.extend([str(v).strip() for v in vals])
    
    # 基础去重和长度过滤
    unique_names = list(set([n for n in all_names if 0 < len(n) < 40]))

    # --- 常见的分子/溶剂黑名单 (Blacklist) ---
    blacklist = [
        # 常见溶剂
        "toluene", "mcp", "dpepo", "dmso", "water", "ethanol", "dichloromethane", 
        "dcm", "chloroform", "methanol", "acetone", "hexane", "thf", "benzene", 
        "acetonitrile", "ethyl acetate", "cyclohexane", "dimethylformamide", "dmf", 
        "pyridine", "propanol", "butanol", "isopropanol", "ipa", "ether", 
        "chlorobenzene", "dichlorobenzene", "mesitylene", "xylenes",
        # 主体材料
        "dpepo", "cbp", "mcp", "mcbp", "ppf", "mcpcn", "ppt", "cztrz", 
        "26dczppy", "tcta", "pbict", "tpbi", "czsi", "pyd2", "o-czoxd", 
        "dbfpo", "mcbp-cn", "dpetpo", "mcppy2po", "simcp2", "bcpo", 
        "mcp-pfp", "sicz", "tmpypb",
        # 聚合物及其他
        "pvk", "pmma", "ps", "mcp-ps", "neat"
    ]

    # 执行过滤逻辑：不区分大小写匹配
    filtered_names = [
        name for name in unique_names 
        if name.lower() not in blacklist
    ]
    
    return filtered_names

# ========= Parallel Identification Task Unit =========
def worker_identify_page(page_img_path, target_names, temp_dir, page_id):
    base64_image = encode_image(page_img_path)
    prompt = """
    You are a chemical structure recognition expert. Please analyze this image:
    
    1. Recognition Criteria:
       - Only recognize 2D chemical structures (ChemDraw style).
       - Strictly no truncated molecules: Bounding box [ymin, xmin, ymax, xmax] must cover everything (substituents, labels).
       - Exclude 3D models or shaded renderings.
    
    2. Extract Information:
       - Provide normalized bbox (0-1000).
       - Score (0.0-1.0) based on completeness.
    
    Strictly return in JSON format:
    {"structures_found": [{"label": "Name", "bbox": [ymin, xmin, ymax, xmax], "score": 0.95}]}
    """
    page_results = []
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        
        for item in data.get("structures_found", []):
            raw_label = str(item.get("label")).strip()
            # 匹配逻辑：AI识别的标签是否在我们的目标列表中
            matched_name = next((t for t in target_names if raw_label.lower() == t.lower()), None)
            
            if matched_name:
                ymin, xmin, ymax, xmax = item["bbox"]
                with Image.open(page_img_path) as img:
                    w, h = img.size
                    l, t, r, b = xmin*w/1000, ymin*h/1000, xmax*w/1000, ymax*h/1000
                    
                    # 增加 10% 边距
                    padding_ratio = 0.1
                    pw = (r - l) * padding_ratio
                    ph = (b - t) * padding_ratio
                    
                    cropped = img.crop((
                        max(0, l - pw), 
                        max(0, t - ph), 
                        min(w, r + pw), 
                        min(h, b + ph)
                    ))
                    
                    cand_name = f"cand_{matched_name}_{page_id}.png"
                    cand_path = os.path.join(temp_dir, cand_name)
                    cropped.save(cand_path, "PNG", optimize=True)
                    
                    page_results.append({
                        "name": matched_name,
                        "path": cand_path,
                        "score": item.get("score", 0),
                        "area": cropped.size[0] * cropped.size[1]
                    })
    except Exception as e:
        print(f" [!] Error processing page {page_id}: {e}")
    finally:
        if os.path.exists(page_img_path): os.remove(page_img_path)
    return page_results

# ========= Main Workflow per DOI =========
def process_doi_workflow(doi, target_names, doi_output_folder, main_pdf_path, si_folder_path):
    """
    修改后的工作流：接收具体的 PDF 路径和 SI 文件夹路径
    """
    start_time = time.time()
    candidate_pool = {}
    
    # 创建临时文件夹
    temp_dir = f"temp_{doi.replace('.', '_').replace('/', '_')}"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 1. 确定 SI PDF 文件
    si_pdf_path = None
    if si_folder_path and os.path.exists(si_folder_path):
        # 简单匹配：文件名包含 DOI 且包含 si 或 support
        clean_doi = doi.replace('/', '-') # 假设文件名里的斜杠被替换了
        for f in os.listdir(si_folder_path):
            if f.lower().endswith('.pdf'):
                # 宽松匹配逻辑，适应不同的命名习惯
                if (doi in f or clean_doi in f) and ("si" in f.lower() or "support" in f.lower()):
                    si_pdf_path = os.path.join(si_folder_path, f)
                    break
    
    task_images = []
    #print(f"--- Targets: {target_names}")
    #print(f"--- Processing: Main={os.path.basename(main_pdf_path)} | SI={os.path.basename(si_pdf_path) if si_pdf_path else 'None'}")

    # 2. 提取 PDF 页面为图片
    pdfs_to_process = [p for p in [main_pdf_path, si_pdf_path] if p and os.path.exists(p)]
    
    for pdf_path in pdfs_to_process:
        try:
            doc = fitz.open(pdf_path)
            pdf_tag = "Main" if pdf_path == main_pdf_path else "SI"
            
            for i in range(len(doc)):
                page = doc[i]
                page_text = page.get_text().lower()
                
                name_match = any(name.lower() in page_text for name in target_names)
                keyword_match = any(kw.lower() in page_text for kw in CONTEXT_KEYWORDS)
                
                if name_match or keyword_match:
                    pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0)) # 3倍够用了，4倍太慢
                    img_path = os.path.join(temp_dir, f"raw_{pdf_tag}_{i}.png")
                    pix.save(img_path)
                    task_images.append((img_path, f"{pdf_tag}_{i}"))
            doc.close()
        except Exception as e:
            print(f" [!] Error opening PDF {pdf_path}: {e}")

    # 3. 多线程调用 LLM 识别结构
    if task_images:
        #print(f"🚀 Sending {len(task_images)} candidate pages to Gemini...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(worker_identify_page, img[0], target_names, temp_dir, img[1]) for img in task_images]
            for future in as_completed(futures):
                results = future.result()
                for res in results:
                    mol_name = res["name"]
                    if mol_name not in candidate_pool: candidate_pool[mol_name] = []
                    candidate_pool[mol_name].append(res)

    # 4. 保存最佳结果
    if candidate_pool:
        for mol_name, cands in candidate_pool.items():
            # 排序：Score > Area
            best = sorted(cands, key=lambda x: (x['score'], x['area']), reverse=True)[0]
            
            # 清理文件名中的非法字符
            safe_mol_name = re.sub(r'[\\/*?:"<>|]', "_", mol_name)
            final_path = os.path.join(doi_output_folder, f"{safe_mol_name}.png")
            
            shutil.copy(best["path"], final_path)
            #print(f" [✓] Saved: {mol_name}")

    # 清理临时文件
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    #print(f"Finished DOI {doi} in {time.time() - start_time:.2f}s")
# ========= Main Entry =========

if __name__ == "__main__":
    
    # 循环处理每个出版社
    for publisher in PUBLISHERS:
        print(f"\n{'='*60}")
        print(f"STARTING PUBLISHER: {publisher}")
        print(f"{'='*60}")

        # 1. 寻找该出版社的 CSV 文件
        # 尝试匹配 {publisher}_TADF_All_Results-DAandMR-cdn.csv 或类似的
        csv_candidates = glob.glob(f"{publisher}_TADF_All_Results*.csv")
        if not csv_candidates:
            print(f"No CSV found for {publisher}. Skipping.")
            continue
        
        csv_source = csv_candidates[0] # 使用找到的第一个
        print(f"Reading CSV: {csv_source}")
        
        try:
            df = pd.read_csv(csv_source)
        except Exception as e:
            print(f"Error reading CSV {csv_source}: {e}")
            continue

        # 2. 准备路径
        # Literatrue/{publisher}
        pub_root_dir = os.path.join(BASE_DIR_NAME, publisher)
        
        # 定义 D-A 和 MR 的文件夹路径
        da_folder = os.path.join(pub_root_dir, "D-A TADF")
        mr_folder = os.path.join(pub_root_dir, "MR-TADF")
        si_folder = os.path.join(pub_root_dir, SI_DIR_NAME)
        
        # 定义输出路径 Literatrue/{publisher}/mol-picture-TADF
        pub_output_folder = os.path.join(pub_root_dir, OUTPUT_FOLDER_NAME)

        if not os.path.exists(pub_root_dir):
            print(f"Publisher folder not found: {pub_root_dir}")
            continue

        # 3. 遍历 CSV 中的 DOI
        # 清理 DOI 列
        df['DOI_Clean'] = df['DOI'].apply(lambda x: str(x).replace(".pdf", "").strip())
        unique_dois = df['DOI_Clean'].unique()
        
        print(f"Found {len(unique_dois)} unique DOIs to process.")

        for current_doi in tqdm(unique_dois, desc=f"{publisher} Progress"):
            # 获取目标分子名
            targets = get_target_names_for_doi(df, current_doi)
            if not targets:
                continue

            # 4. 定位 PDF 文件 (优先 D-A，其次 MR)
            # 假设文件名就是 {doi}.pdf 或者 {doi} (需要处理扩展名)
            # 为了稳健，我们检查加 .pdf 和不加的情况
            
            pdf_name = f"{current_doi}.pdf" if not current_doi.lower().endswith(".pdf") else current_doi
            
            found_main_pdf = None
            
            # 检查 D-A TADF
            path_da = os.path.join(da_folder, pdf_name)
            # 检查 MR-TADF
            path_mr = os.path.join(mr_folder, pdf_name)
            
            # 有些文件夹可能有子文件夹，这里简化处理，直接查指定路径
            # 你的上一步代码是在 pass-DAandMR 里，但 prompt 说是在 Literatrue/{pub}/D-A TADF 下
            # 如果文件名不匹配，可能需要模糊搜索，这里先按精准文件名匹配
            
            if os.path.exists(path_da):
                found_main_pdf = path_da
            elif os.path.exists(path_mr):
                found_main_pdf = path_mr
            
            if not found_main_pdf:
                # 尝试去除非法字符再次匹配 (有些文件名把 / 换成了 -)
                safe_name = current_doi.replace("/", "-")
                if not safe_name.endswith(".pdf"): safe_name += ".pdf"
                
                path_da_safe = os.path.join(da_folder, safe_name)
                path_mr_safe = os.path.join(mr_folder, safe_name)
                
                if os.path.exists(path_da_safe):
                    found_main_pdf = path_da_safe
                elif os.path.exists(path_mr_safe):
                    found_main_pdf = path_mr_safe
            
            if not found_main_pdf:
                # print(f"PDF not found for DOI {current_doi}. Skipping.")
                continue

            # 5. 设置 DOI 输出目录
            # Literatrue/{publisher}/mol-picture-TADF/{doi_clean}
            doi_subfolder_name = current_doi.replace("/", "_")
            doi_output_path = os.path.join(pub_output_folder, doi_subfolder_name)
            
            # 如果该 DOI 已经处理过且有结果，是否跳过？(这里不跳过，覆盖模式)
            os.makedirs(doi_output_path, exist_ok=True)
            
            # 6. 执行提取工作流
            #print(f"\nProcessing: {current_doi}")
            process_doi_workflow(current_doi, targets, doi_output_path, found_main_pdf, si_folder)

    print(f"\n=== All images have been extracted ===")