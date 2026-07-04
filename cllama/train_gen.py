import os
import re
import sys
import subprocess
from datetime import datetime


# ============================================================
# Script configuration
# ============================================================
DATA_GEN_SCRIPT = os.path.join("module", "gen_data.py")

TRAIN_SUBMIT_TEMPLATE = "python_train_gen.py"
TRAIN_SCRIPT = "CLLaMa_enhanced10.py"
MASK_GEN_SCRIPT = "gen_est_049_mask.py"

LOG_FILE = "Initialize_gen.log"
JOB_ID_RECORD_FILE = "last_training_job_id.txt"
RUNTIME_SLURM_DIR = "slurm_generation_training"

DEFAULT_PARTITION = os.environ.get("FIREFLY_GEN_TRAIN_PARTITION", "gpu")
DEFAULT_NODE = os.environ.get("FIREFLY_GEN_TRAIN_NODE", "gpu1")
DEFAULT_NTASKS = os.environ.get("FIREFLY_GEN_TRAIN_NTASKS", "10")
DEFAULT_GRES = os.environ.get("FIREFLY_GEN_TRAIN_GRES", "")

SUBMITTED_SLURM_JOB_IDS = []


def write_log(message):
    """Print a message and write it to the log file."""
    print(message, flush=True)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def check_script_exists(script_name):
    """Check whether the target script exists."""
    if not os.path.exists(script_name):
        write_log("Required workflow component not found.")
        write_log(f"Missing file: {script_name}")
        write_log("Please check the working directory and file structure.")
        sys.exit(1)


def ask_yes_no(question, default="n"):
    """
    Ask a yes/no question.

    default = "y" means pressing Enter returns True.
    default = "n" means pressing Enter returns False.
    """
    default = default.lower()

    if default not in ["y", "n"]:
        raise ValueError("Default value must be 'y' or 'n'.")

    while True:
        answer = input(f"{question} [y/n, default: {default}]: ").strip().lower()

        if answer == "":
            return default == "y"

        if answer in ["y", "yes"]:
            return True

        if answer in ["n", "no"]:
            return False

        print("Invalid input. Please enter y or n.")


def ask_text(prompt, default=""):
    """Ask for a text value with an optional default."""
    if default:
        answer = input(f"{prompt} [default: {default}]: ").strip()
        return answer if answer else default

    answer = input(f"{prompt} [press Enter to skip]: ").strip()
    return answer


def sanitize_label(value):
    """Make a safe label fragment for filenames."""
    value = str(value).strip()
    if not value:
        return "auto"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def run_python_script(script_name, task_name):
    """
    Run a Python script in the current terminal.

    Interactive input is preserved. If the target task requests input,
    the user can type it directly in the terminal.
    """
    check_script_exists(script_name)

    write_log("=" * 80)
    write_log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start: {task_name}")
    write_log("=" * 80)

    result = subprocess.run([sys.executable, script_name])

    write_log("=" * 80)
    write_log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] End: {task_name}")
    write_log(f"Exit code: {result.returncode}")
    write_log("=" * 80)

    if result.returncode != 0:
        write_log(f"Execution failed during: {task_name}")
        write_log("Workflow terminated.")
        sys.exit(result.returncode)


def collect_slurm_options():
    """Collect SLURM partition/node options for generative model training."""
    print("\nSLURM resource settings for generative-model training")
    print("-" * 80)
    print("Examples:")
    print("  partition: gpu, node: gpu1")
    print("  partition: gpu, node: gpu2")
    print("  partition: cpu, node: <press Enter for automatic allocation>")
    print("-" * 80)

    partition = ask_text("Enter SLURM partition name", DEFAULT_PARTITION)
    node = ask_text("Enter node name, e.g., gpu1 or gpu2", DEFAULT_NODE)
    ntasks = ask_text("Enter number of CPU tasks", DEFAULT_NTASKS)
    gres = ask_text("Enter optional GPU GRES, e.g., gpu:1", DEFAULT_GRES)

    return {
        "partition": partition,
        "node": node,
        "ntasks": ntasks,
        "gres": gres,
    }


def create_runtime_slurm_script(options):
    """Create a runtime SLURM script using user-selected resources."""
    check_script_exists(TRAIN_SCRIPT)
    os.makedirs(RUNTIME_SLURM_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    partition_label = sanitize_label(options.get("partition"))
    node_label = sanitize_label(options.get("node"))
    script_name = f"submit_gen_train_{partition_label}_{node_label}_{timestamp}.sh"
    script_path = os.path.join(RUNTIME_SLURM_DIR, script_name)

    partition = options.get("partition", "").strip()
    node = options.get("node", "").strip()
    ntasks = options.get("ntasks", "").strip() or DEFAULT_NTASKS
    gres = options.get("gres", "").strip()

    lines = [
        "#!/bin/bash",
        "#SBATCH -J fg_gen_train",
    ]

    if partition:
        lines.append(f"#SBATCH -p {partition}")
    if node:
        lines.append(f"#SBATCH -w {node}")

    lines.extend([
        "#SBATCH -N 1",
        f"#SBATCH -n {ntasks}",
    ])

    if gres:
        lines.append(f"#SBATCH --gres={gres}")

    lines.extend([
        f"#SBATCH --error={RUNTIME_SLURM_DIR}/%J.err",
        f"#SBATCH --output={RUNTIME_SLURM_DIR}/%J.out",
        "",
        "export I_MPI_ADJUST_REDUCE=3",
        "cd $SLURM_SUBMIT_DIR",
        "",
        f"python {TRAIN_SCRIPT}",
        "",
    ])

    with open(script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))

    return script_path


def submit_slurm_job(submit_script, task_name):
    """
    Submit a SLURM job using sbatch and return the job ID.

    This function only submits the job.
    It does not wait for the job to finish.
    It does not cancel the job when this workflow exits.
    """
    check_script_exists(submit_script)

    write_log("=" * 80)
    write_log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Submit: {task_name}")
    write_log(f"SBATCH script: {submit_script}")
    write_log("=" * 80)

    result = subprocess.run(
        ["sbatch", submit_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        write_log(f"Failed to submit SLURM job for: {task_name}")
        write_log(result.stderr.strip())
        write_log("Workflow terminated.")
        sys.exit(result.returncode)

    stdout = result.stdout.strip()
    write_log(stdout)

    job_id = None
    for token in stdout.split():
        if token.isdigit():
            job_id = token
            break

    if job_id is None:
        write_log("Failed to parse SLURM job ID from sbatch output.")
        write_log("Workflow terminated.")
        sys.exit(1)

    SUBMITTED_SLURM_JOB_IDS.append(job_id)

    with open(JOB_ID_RECORD_FILE, "w", encoding="utf-8") as f:
        f.write(f"{job_id}\n")

    write_log(f"SLURM job submitted successfully. Job ID: {job_id}")
    write_log(f"Job ID has been saved to: {JOB_ID_RECORD_FILE}")

    return job_id


def print_submitted_job_note(job_id):
    """Print a clear note after submitting a SLURM job."""
    print("\nTraining job has been submitted successfully.")
    print(f"SLURM job ID: {job_id}")
    print("This training job is now managed by SLURM.")
    print("Stopping this workflow will not cancel the submitted SLURM job.")
    print(f"The job ID has been saved to: {JOB_ID_RECORD_FILE}")
    print("\nUseful commands:")
    print(f"  squeue -j {job_id}")
    print(f"  tail -f {RUNTIME_SLURM_DIR}/{job_id}.out")
    print(f"  tail -f {RUNTIME_SLURM_DIR}/{job_id}.err")
    print(f"  scancel {job_id}    # only if you really want to cancel the job")


def main():
    write_log("\n")
    write_log("#" * 80)
    write_log(f"Firefly-Geni workflow started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_log("#" * 80)

    # ========================================================
    # Stage 1: dataset preprocessing
    # ========================================================
    print("\nStage 1/3: Dataset preprocessing")
    print("This step regenerates the fixed token_dataset used by downstream tasks.")
    print("The augmentation multiplier will be requested during preprocessing.")
    print("Use multiplier = 1 for no augmentation.")

    run_data_gen = ask_yes_no(
        "Regenerate token_dataset?",
        default="y"
    )

    if run_data_gen:
        write_log("Stage 1 selected: regenerate token_dataset.")
        write_log("The augmentation multiplier will be requested during preprocessing.")
        write_log("Use multiplier = 1 for no augmentation.")

        run_python_script(
            script_name=DATA_GEN_SCRIPT,
            task_name="Dataset preprocessing"
        )
    else:
        write_log("Stage 1 skipped: reuse the existing token_dataset.")

    # ========================================================
    # Stage 2: generative model training
    # ========================================================
    print("\nStage 2/3: Generative model training")
    print("This step submits the generative model training job to SLURM.")
    print("The workflow will not wait for the training job to finish.")
    print("After submission, the SLURM job will continue independently.")

    run_train = ask_yes_no(
        "Submit generative model training job?",
        default="y"
    )

    training_job_id = None

    if run_train:
        write_log("Stage 2 selected: submit generative model training job.")

        slurm_options = collect_slurm_options()
        runtime_script = create_runtime_slurm_script(slurm_options)

        write_log("Runtime SLURM script created:")
        write_log(runtime_script)
        write_log("Selected SLURM resources:")
        for key, value in slurm_options.items():
            write_log(f"  {key}: {value if value else '<not set>'}")

        training_job_id = submit_slurm_job(
            submit_script=runtime_script,
            task_name="Generative model training"
        )

        write_log("Training job has been submitted successfully.")
        write_log(f"SLURM job ID: {training_job_id}")
        write_log("The workflow will not wait for the training job to finish.")
        write_log("Stopping this workflow will not cancel the submitted SLURM job.")
        write_log("Use scancel manually only if the submitted job should be cancelled.")

        print_submitted_job_note(training_job_id)

        stop_after_submit = ask_yes_no(
            "Stop this workflow after submitting the training job?",
            default="y"
        )

        if stop_after_submit:
            write_log("Workflow stopped after submitting the training job.")
            write_log(f"The submitted SLURM job remains active. Job ID: {training_job_id}")
            write_log("#" * 80)
            write_log("Firefly-Geni workflow completed.")
            write_log("#" * 80)
            sys.exit(0)

    else:
        write_log("Stage 2 skipped: generative model training was not submitted.")

    # ========================================================
    # Stage 3: mask-based molecular generation
    # ========================================================
    print("\nStage 3/3: Mask-based molecular generation")
    print("This step performs property-guided molecular generation.")
    print("Target EST, target SA, and generation size will be requested during generation.")

    if training_job_id is not None:
        print("\nNote:")
        print("A new training job has just been submitted but may not be finished yet.")
        print("Continuing now will use the currently available checkpoint.")
        print("It will not wait for the newly submitted training job.")

        run_mask_generation = ask_yes_no(
            "Continue to mask-based molecular generation now?",
            default="n"
        )
    else:
        run_mask_generation = ask_yes_no(
            "Execute mask-based molecular generation?",
            default="y"
        )

    if run_mask_generation:
        write_log("Stage 3 selected: execute mask-based molecular generation.")
        write_log("Target EST, target SA, and generation size will be requested during generation.")

        run_python_script(
            script_name=MASK_GEN_SCRIPT,
            task_name="Mask-based molecular generation"
        )
    else:
        write_log("Stage 3 skipped: mask-based molecular generation was not executed.")

        if training_job_id is not None:
            write_log(f"The submitted training job remains active. Job ID: {training_job_id}")
            write_log("After training is completed, rerun this workflow and skip training to perform generation.")

    write_log("#" * 80)
    write_log("Firefly-Geni workflow completed.")
    write_log("#" * 80)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        write_log("\nWorkflow interrupted by user.")

        if len(SUBMITTED_SLURM_JOB_IDS) > 0:
            write_log("Submitted SLURM jobs were not cancelled by this workflow.")
            write_log("Submitted SLURM job IDs:")
            for job_id in SUBMITTED_SLURM_JOB_IDS:
                write_log(f"  {job_id}")
            write_log("Use scancel manually only if a submitted job should be cancelled.")
        else:
            write_log("No SLURM job was recorded as submitted by this workflow.")

        sys.exit(130)
