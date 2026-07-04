import subprocess
import sys
import time
import os


def format_elapsed_time(seconds):
    """Format elapsed seconds as HH:MM:SS."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def run_step(script_name, success_keyword, step_index, total_steps):
    """
    运行单个脚本，并检查输出中是否包含成功关键词。
    """
    print("=" * 60)
    print(f"🚀 [Step {step_index}/{total_steps}] Running: {script_name}...")
    print("=" * 60)

    # 记录开始时间
    start_time = time.time()
    
    # 捕获所有输出内容用于后续检查
    full_output = []
    
    # 使用 Popen 启动子进程，实时获取输出。
    # Task 1 is launched from the LLM_Extract environment by auto_Firefly-Geni.py.
    # Therefore sys.executable keeps all normal data-extraction steps in LLM_Extract.
    child_env = os.environ.copy()
    child_env.setdefault("PYTHONNOUSERSITE", "1")
    # Force child scripts to emit UTF-8 on Windows. Some LLM/PDF tools may
    # otherwise write GBK/ANSI bytes to the pipe, which can crash this
    # controller while reading real-time output.
    child_env.setdefault("PYTHONIOENCODING", "utf-8")
    child_env.setdefault("PYTHONUTF8", "1")
    process = subprocess.Popen(
        [sys.executable, script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, # 将错误输出也合并到标准输出中
        text=True,
        encoding='utf-8',  # Decode child output as UTF-8.
        errors='replace',   # Never stop the pipeline because of one undecodable byte.
        bufsize=1,
        env=child_env,
    )

    # 实时打印输出到控制台，并保存到 full_output 列表
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line.strip()) # 实时显示
            full_output.append(line)

    # 等待进程完全结束
    return_code = process.poll()
    end_time = time.time()
    duration = end_time - start_time

    # 将所有输出合并成一个字符串用于查找
    output_text = "".join(full_output)

    # 核心检查逻辑
    if success_keyword in output_text:
        print("\n" + "-" * 60)
        print(f"✅ Step {step_index} Success: {script_name}")
        print(f"   Key phrase found: '{success_keyword}'")
        print(f"   Time taken: {duration:.2f} seconds")
        print("-" * 60 + "\n")
        return True
    else:
        print("\n" + "!" * 60)
        print(f"❌ Step {step_index} Failed: {script_name}")
        print(f"   Missing key phrase: '{success_keyword}'")
        print("   Pipeline stopped due to error or incomplete execution.")
        print("!" * 60 + "\n")
        return False

def main():
    total_start_time = time.time()
    # 定义流程配置列表
    # 格式: (脚本文件名, 判定成功的关键词)
    pipeline_steps = [
        ("Classification.py", "=== The classification has been completed ==="),
        ("Data_Extraction.py", "=== All publishers processed. Data saved ==="),
        ("Molecular_image_extraction.py", "=== All images have been extracted ==="),
        ("Image_reconstruction.py", "=== All images have been reconstructed ==="),
        ("Graph2smiles_env.py", "=== SMILES conversion completed successfully ==="),
        ("1_smiles_process.py", "Completed standardization and removed all chirality information"),
        ("2_data_process.py", "Integration and sorting complete!"),
        ("3_solvent_process.py", "Solution information processed successfully"),
        ("4_dataset_process.py", "The database has been fully organized"),
    ]

    total_steps = len(pipeline_steps)
    print(f"🤖 Automation started. Total steps: {total_steps}\n")

    # 循环执行每一步
    for i, (script, keyword) in enumerate(pipeline_steps, 1):
        # 检查文件是否存在
        if not os.path.exists(script):
            print(f"❌ Error: Script file not found: {script}")
            sys.exit(1)
            
        # 执行脚本
        success = run_step(script, keyword, i, total_steps)
        
        # 如果失败，直接退出程序
        if not success:
            sys.exit(1)

    # 全部完成后的最终提示
    print("=" * 60)
    print("🎉🎉🎉 CONGRATULATIONS! 🎉🎉🎉")
    print("All data extraction, SMILES processing, and database organization tasks are complete.")
    print("=" * 60)
    
    total_elapsed = time.time() - total_start_time
    print(f"Total elapsed time: {format_elapsed_time(total_elapsed)}")

if __name__ == "__main__":
    main()