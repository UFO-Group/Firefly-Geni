import base64
import os
import requests
import re
import time
import sys
from openai import OpenAI
from tqdm import tqdm

# ========= 1. Path & API Setup =========
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import API config
try:
    from LLMs_API import client, get_model_name
except ImportError:
    print("Error: Could not find LLMs_API.py in the parent directory.")
    sys.exit(1)

# ========= 2. Model Configuration =========
GEN_MODEL = get_model_name("image")  
SCORE_MODEL = get_model_name("flash") 

print(f"Generative Model: {GEN_MODEL}")
print(f"Scoring Model:    {SCORE_MODEL}")

# ========= 3. Path Configuration (Modified) =========
BASE_DIR_NAME = "Literatrue" 
OUTPUT_FOLDER_NAME = "mol-picture-TADF" 
PUBLISHERS = ["acs", "elsevier", "nature", "rsc", "wiley"]

PROXIES = None 
MAX_RETRIES = 3

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_similarity_score(original_b64, reconstructed_b64):
    """
    调用模型对比原图和重构图，给出相似度分数
    """
    score_prompt = """
    Analyze two images: (1) original chemical structure image, (2) reconstructed ChemDraw-style image.
    Task: Rate the similarity/accuracy from 0.00 to 1.00.
    
    SCORING PRIORITY (must follow):
    Step A — Cleanliness gate (HARD RULE):
    - If ANY text/letters/numbers remain OR any non-target molecule fragments exist OR any stray/extra lines/artifacts exist,
      then the score MUST be <= 0.70 (cap at 0.70), regardless of connectivity.
    
    Step B — Chemical correctness:
    - If any atom is missing, any bond order is wrong, any ring closure is wrong, or B/N placement is incorrect,
      then the score MUST be <= 0.50 (cap at 0.50).
    
    Step C — Geometry/style:
    - If connectivity is correct and no artifacts/text remain, score based on geometric similarity (bond angles/relative layout),
      line sharpness, pure black lines, pure white background.
    
    Score bands:
    - 0.95–1.00: Perfect connectivity AND geometry close match AND absolutely clean (no text, no extra lines, no fragments).
    - 0.80–0.94: Correct connectivity, minor geometry/style deviations, BUT still perfectly clean.
    - 0.50–0.79: Any artifacts/text/extra lines/fragments present (even if connectivity is correct).
    - 0.00–0.49: Incorrect connectivity, missing atoms, wrong bonding, wrong B/N placement, or major structural errors.
    
    Output ONLY the numerical score with two decimals (e.g., 0.98). No other text.

    """
    try:
        response = client.chat.completions.create(
            model=SCORE_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": score_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{original_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{reconstructed_b64}"}}
                ],
            }],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        match = re.search(r'0\.\d+|1\.0', content)
        return float(match.group(0)) if match else 0.0
    except Exception as e:
        #print(f"    [!] 评分系统故障: {e}")
        return 0.0

def save_image_from_content(content, temp_path):
    """
    从生成模型的回复中保存图片
    """
    base64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
    b64_match = re.search(base64_pattern, content)
    if b64_match:
        try:
            img_data = base64.b64decode(b64_match.group(1))
            with open(temp_path, 'wb') as f:
                f.write(img_data)
            return True
        except: pass

    urls = re.findall(r'https?://[^\s\)\!\]\"\'<>]+', content)
    for url in urls:
        if any(url.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp']):
            try:
                res = requests.get(url, stream=True, timeout=60, proxies=PROXIES)
                if res.status_code == 200:
                    with open(temp_path, 'wb') as f:
                        for chunk in res.iter_content(1024): f.write(chunk)
                    return True
            except: pass
    return False

def process_image(image_path, doi_output_folder):
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    temp_save_path = os.path.join(doi_output_folder, f"reconstructed_{base_filename}_temp.png")
    orig_b64 = encode_image(image_path)
    
    gen_prompt = """
    You are a professional chemical structure illustrator.
    Task: Analyze the uploaded image and generate a clean ChemDraw-style drawing of the target molecule.
    
    Rules (follow in order):
    1. Select the largest molecular structure in the center of the image as the target molecule.
    2. Edit the original image only: remove/cover all text (labels, legend letters, titles, background text). 
    3. Delete any non-target molecular fragments; do not add any content.
    4. Remove any non-target molecular fragments; do not add or invent anything.
    5. ChemDraw-style output: pure black lines on a pure white background; preserve the exact connectivity/geometry as seen in the input.
    6. Output requirement: generate the image using your internal tool.
    7. Provide the image link only. No text output.
    """

    attempt = 0
    final_score = 0.0
    success = False

    while attempt < MAX_RETRIES:
        attempt += 1
        #print(f"  -> [{base_filename}] 尝试绘图 第 {attempt} 次...")
        
        try:
            # 1. 生成图片
            response = client.chat.completions.create(
                model=GEN_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": gen_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{orig_b64}"}}
                    ],
                }]
            )
            
            content = response.choices[0].message.content
            if save_image_from_content(content, temp_save_path):
                # 2. 评分
                recon_b64 = encode_image(temp_save_path)
                final_score = get_similarity_score(orig_b64, recon_b64)
                #print(f"    当前评分: {final_score}")

                if final_score >= 0.95:
                    #print(f"    [✓] 分数达标 (>=0.95)，停止重试。")
                    success = True
                    break
                else:
                    #print(f"    [!] 分数不达标 (<0.95)，准备重新生成...")
                    pass
            else:
                #print(f"    [!] 绘图模型未返回有效图片。")
                pass
        except Exception as e:
            #print(f"    [!] 运行异常: {e}")
            pass
        
        time.sleep(2) # 避免请求过快

    # 3. 循环结束，保存最终结果
    if os.path.exists(temp_save_path):
        final_name = os.path.join(doi_output_folder, f"reconstructed_{base_filename}_score_{final_score}.png")
        if os.path.exists(final_name): os.remove(final_name)
        os.rename(temp_save_path, final_name)
        #print(f"    [✓] 最终文件已存: {os.path.basename(final_name)}")
    else:
        #print(f"    [X] {base_filename} 处理彻底失败。")
        pass

def batch_process_mol_pictures():
    # Loop through all publishers
    for publisher in PUBLISHERS:
        # Construct the image path for the current publisher: Literatrue/{publisher}/mol-picture-TADF
        input_root_dir = os.path.join(BASE_DIR_NAME, publisher, OUTPUT_FOLDER_NAME)
        
        print(f"\n{'='*60}")
        print(f"Processing publisher: {publisher}")
        print(f"Path: {input_root_dir}")
        print(f"{'='*60}")

        if not os.path.exists(input_root_dir):
            print(f"  [Skip] Directory does not exist: {input_root_dir}")
            continue

        doi_folders = [f for f in os.listdir(input_root_dir) if os.path.isdir(os.path.join(input_root_dir, f))]
        
        if not doi_folders:
            print(f"  [Info] No DOI folders found in the directory.")
            continue
            
        # --- Pre-scan phase: Collect all tasks for this publisher ---
        # Purpose: To display an accurate total progress bar
        all_tasks = []
        print("  Scanning for images to process...")
        for doi in doi_folders:
            doi_path = os.path.join(input_root_dir, doi)
            doi_output_folder = os.path.join(doi_path, "reconstructed_with_scores")
            
            # Ensure the output directory exists
            if not os.path.exists(doi_output_folder):
                os.makedirs(doi_output_folder)
            
            # Find all image files (excluding already generated 'reconstructed' files)
            images = [
                f for f in os.listdir(doi_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) 
                and not f.startswith("reconstructed_")
            ]
            
            for img_name in images:
                all_tasks.append({
                    "img_path": os.path.join(doi_path, img_name),
                    "output_folder": doi_output_folder
                })

        if not all_tasks:
            print("  No images found to process.")
            continue

        print(f"  Found {len(all_tasks)} images, starting reconstruction...")

        # --- Execution phase: With progress bar ---
        # Use tqdm to display progress
        for task in tqdm(all_tasks, desc=f"{publisher} Progress"):
            process_image(task["img_path"], task["output_folder"])
            # Simple delay to prevent excessive concurrency
            time.sleep(0.5)

    print(f"\n=== All images have been reconstructed ===")

if __name__ == "__main__":
    batch_process_mol_pictures()