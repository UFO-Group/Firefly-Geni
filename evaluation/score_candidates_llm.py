# -*- coding: utf-8 -*-
"""
score_candidates_llm.py

Recommended location:
    Firefly-Geni/evaluation/score_candidates_llm.py

Run:
    cd Firefly-Geni/evaluation
    python score_candidates_llm.py

Purpose:
    Score candidate TADF molecules in three emission-wavelength folders using
    the LLM API client defined in:

        Firefly-Geni/LLMs_API.py

Input files:
    Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/emission_380_495/
        molecules_emission_380_495.csv

    Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/emission_495_570/
        molecules_emission_495_570.csv

    Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/emission_570_770/
        molecules_emission_570_770.csv

Output files:
    Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/emission_380_495/
        molecules_emission_380_495_score.csv

    Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/emission_495_570/
        molecules_emission_495_570_score.csv

    Firefly-Geni/evaluation/candidate/gen_from_iter4_epoch049_mask/emission_570_770/
        molecules_emission_570_770_score.csv
"""

from pathlib import Path
import os
import json
import time
import re
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd


# =======================
# 1. Project paths
# =======================

# This script is expected to be placed in:
# Firefly-Geni/evaluation/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Firefly-Geni/LLMs_API.py is at project root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =======================
# 2. API Configuration
# =======================

# Do not define API keys here. They are loaded from Firefly-Geni/LLMs_API.py.
from LLMs_API import client, MODEL_PRO

MODEL_NAME = MODEL_PRO

if client is None:
    raise RuntimeError(
        "LLM API client is None. Please check Firefly-Geni/LLMs_API.py."
    )

# Number of concurrent independent API requests.
MAX_WORKERS = 12


# =======================
# 3. Input and Output Files
# =======================

# Selected screening folder. If FIREFLY_SCREENING_CANDIDATE_DIR is not set, use the legacy root folder.
CANDIDATE_DIR = Path(os.environ.get("FIREFLY_SCREENING_CANDIDATE_DIR", str(SCRIPT_DIR / "candidate" / "gen_from_iter4_epoch049_mask"))).expanduser().resolve()

INPUT_OUTPUT_PAIRS = [
    (
        CANDIDATE_DIR / "emission_380_495" / "molecules_emission_380_495.csv",
        CANDIDATE_DIR / "emission_380_495" / "molecules_emission_380_495_score.csv",
    ),
    (
        CANDIDATE_DIR / "emission_495_570" / "molecules_emission_495_570.csv",
        CANDIDATE_DIR / "emission_495_570" / "molecules_emission_495_570_score.csv",
    ),
    (
        CANDIDATE_DIR / "emission_570_770" / "molecules_emission_570_770.csv",
        CANDIDATE_DIR / "emission_570_770" / "molecules_emission_570_770_score.csv",
    ),
]


# =======================
# 4. LLM Scoring Prompt
# =======================

SYSTEM_PROMPT = """
You are a leading expert in organic luminescent materials and TADF molecular design. Based on the given molecular SMILES and machine-learning-predicted properties, rigorously evaluate the chemical rationality of the molecule as a TADF candidate material.

Please strictly score the molecule from the following five aspects. Each score must be from 0 to 10 and may be given with one decimal place, where 10 represents the most ideal case:

1. TADF_structure_score, structural and mechanistic compatibility:
Based on the given ΔEST and other predicted properties, evaluate whether the molecular framework is consistent with established TADF design principles, including D–A, D–A–D, D–π–A, MR-TADF rigid frameworks, and related architectures. If the structure conforms to these principles, assign a score of 6 or higher. If the predicted ΔEST is extremely low but the SMILES structure lacks clear orbital separation, multi-resonance characteristics, or a reasonable charge-transfer pathway, this score should not exceed 6.

2. stability_score, chemical stability:
Evaluate the chemical stability of the structure. Structures containing abnormal valence states, highly distorted rings, fragile or unstable bonds such as O–O and N–N, overly reactive functional groups, clearly unstable heteroatom connections, or unprotected highly reactive sites should receive a score no higher than 6. Stable and rigid aromatic systems, reasonable heteroaromatic frameworks, and well-defined conjugated backbones should receive a score of 6 or higher.

3. scaffold_novelty_score, out-of-distribution scaffold novelty:
The input sim_max must be considered; a lower sim_max indicates greater dissimilarity from the training set. If sim_max is low and the structure shows clear innovation in the donor, acceptor, or connection mode, a score of 8 or higher may be assigned. If the molecule exhibits evident structural innovation while retaining TADF-relevant structural features, this score may also be 8 or higher. If the novelty is moderate but some minor innovation is present, the score may range from 6 to 8. If the molecule is merely a simple substitution, homologous extension, or minor modification of a traditional TADF scaffold, this score should be 6 or lower, even if the predicted properties are favorable.

4. fragment_novelty_score, donor/acceptor fragment novelty:
First, attempt to identify the main donor fragments, acceptor fragments, and possible bridging fragments from the SMILES. Determine whether these fragments belong to very common TADF motifs, such as carbazole, dimethylacridine, phenoxazine, phenothiazine, triazine, diphenyl sulfone, benzonitrile, boron–nitrogen multi-resonance frameworks, or related motifs. If either the donor or acceptor fragment has structural features clearly distinct from common TADF fragments, assign a score of 10. If fragment-level novelty is present but not particularly high, assign a score above 8. It is necessary to assess whether the newly identified fragment has previously appeared in TADF molecules; if it has been reported or is commonly used in TADF systems, assign a score between 6 and 8. If the structure is only a simple combination, positional variation, or minor substitution of common donors and common acceptors, assign a score of 6 or lower. Note that this score focuses on fragment-level novelty rather than the overall Tanimoto similarity.

5. environment_performance_score, performance across different environments:
Compare the predicted PLQY values in toluene and DPEPO. Evaluate the molecule’s predicted performance across different environments. If the PLQY is higher than 60% in both environments, assign a score of 9 or higher. If only the PLQY in DPEPO is higher than 60%, assign a score of 8. If only the PLQY in DPEPO falls within the range of 25–60%, assign a score between 6 and 8. If the PLQY is lower than 25% in both environments, assign a score below 6.

Strictly output only the following JSON format. Do not include Markdown code blocks or any additional explanatory text:

{
  "TADF_structure_score": 0.0,
  "stability_score": 0.0,
  "scaffold_novelty_score": 0.0,
  "fragment_novelty_score": 0.0,
  "environment_performance_score": 0.0,
  "reason": "An expert comment within 100 English characters, briefly explaining the donor/acceptor features, fragment novelty, environmental performance, and core scoring rationale."
}
"""


def build_user_prompt(row):
    return f"""
Input information:
SMILES: {row["SMILES"]}
SA_score: {row["SA_score"]}

toluene_absorption_nm: {row["toluene_absorption_nm"]}
toluene_emission_nm: {row["toluene_emission_nm"]}
toluene_Delta_EST_eV: {row["toluene_Delta_EST_eV"]}
toluene_PLQY_percent: {row["toluene_PLQY_percent"]}

dpepo_absorption_nm: {row["dpepo_absorption_nm"]}
dpepo_emission_nm: {row["dpepo_emission_nm"]}
dpepo_Delta_EST_eV: {row["dpepo_Delta_EST_eV"]}
dpepo_PLQY_percent: {row["dpepo_PLQY_percent"]}

sim_max_to_training_set: {row["sim_max"]}
"""


# =======================
# 5. Time Functions
# =======================

def current_time_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_seconds(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# =======================
# 6. JSON Parsing Functions
# =======================

def extract_json(text):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.S)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"Failed to parse JSON: {text}")


def validate_score(value):
    try:
        value = round(float(value), 1)
    except Exception:
        value = 0.0

    return max(0.0, min(10.0, value))


# =======================
# 7. Output Table Formatting Function
# =======================

def prepare_output_df(df):
    """
    Format the output table before saving:
    1. Remove per-molecule API timing columns.
    2. Sort by source_index.
    3. Remove duplicated source_index rows.
    4. Move reason to the last column.
    """
    if df is None or len(df) == 0:
        return df

    df = df.copy()

    drop_cols = [
        "api_time_seconds",
        "api_time_formatted",
        "overall_recommendation",
        "synthesis_score",
        "environment_robustness_score",
        "low_risk_score"
    ]

    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    if "source_index" in df.columns:
        df = df.drop_duplicates(subset="source_index", keep="last")
        df = df.sort_values("source_index")

    if "reason" in df.columns:
        cols = [col for col in df.columns if col != "reason"]
        cols.append("reason")
        df = df[cols]

    return df


# =======================
# 8. Single-Molecule Scoring Function
# =======================

def score_one_molecule(row, max_retries=3, sleep_time=2):
    user_prompt = build_user_prompt(row)

    for attempt in range(1, max_retries + 1):
        try:
            api_start_time = time.time()

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )

            api_elapsed = time.time() - api_start_time

            content = response.choices[0].message.content
            result = extract_json(content)

            result["TADF_structure_score"] = validate_score(
                result.get("TADF_structure_score", 0)
            )
            result["stability_score"] = validate_score(
                result.get("stability_score", 0)
            )
            result["scaffold_novelty_score"] = validate_score(
                result.get("scaffold_novelty_score", 0)
            )
            result["fragment_novelty_score"] = validate_score(
                result.get("fragment_novelty_score", 0)
            )
            result["environment_performance_score"] = validate_score(
                result.get("environment_performance_score", 0)
            )

            if "reason" not in result:
                result["reason"] = ""

            # LLM overall score, ranging from 0 to 10.
            # This weighting emphasizes scaffold novelty and donor/acceptor fragment novelty.
            result["S_LLM"] = (
                0.20 * result["TADF_structure_score"]
                + 0.15 * result["stability_score"]
                + 0.23 * result["scaffold_novelty_score"]
                + 0.27 * result["fragment_novelty_score"]
                + 0.15 * result["environment_performance_score"]
            )

            result["S_LLM"] = round(result["S_LLM"], 3)

            result["api_time_seconds"] = round(api_elapsed, 2)
            result["api_time_formatted"] = format_seconds(api_elapsed)

            return result

        except Exception as e:
            wait_time = sleep_time * attempt + random.uniform(0, 1)

            print("----------------------------------------")
            print("API call failed")
            print(f"Failure time: {current_time_str()}")
            print(f"Attempt {attempt} failed: {e}")
            print(f"Waiting {wait_time:.1f} seconds before retrying")
            print("----------------------------------------")

            time.sleep(wait_time)

    return {
        "TADF_structure_score": None,
        "stability_score": None,
        "scaffold_novelty_score": None,
        "fragment_novelty_score": None,
        "environment_performance_score": None,
        "S_LLM": None,
        "api_time_seconds": None,
        "api_time_formatted": None,
        "reason": "API call or JSON parsing failed"
    }


# =======================
# 9. Parallel CSV Scoring Function
# =======================

def score_csv_parallel(input_file, output_file, save_every=20, max_workers=6):
    file_start_time = time.time()

    input_file = Path(input_file)
    output_file = Path(output_file)

    print("========================================")
    print(f"Start processing file: {input_file}")
    print(f"Start time: {current_time_str()}")
    print(f"Output file: {output_file}")
    print(f"Model name: {MODEL_NAME}")
    print(f"Number of concurrent requests: {max_workers}")
    print("========================================")

    if not input_file.exists():
        print(f"Input file not found, skip: {input_file}")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)

    required_cols = [
        "SMILES",
        "SA_score",
        "toluene_absorption_nm",
        "toluene_emission_nm",
        "toluene_Delta_EST_eV",
        "toluene_PLQY_percent",
        "dpepo_absorption_nm",
        "dpepo_emission_nm",
        "dpepo_Delta_EST_eV",
        "dpepo_PLQY_percent",
        "sim_max"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in {input_file}: {col}")

    df = df.reset_index(drop=True)
    df["source_index"] = df.index

    results = []
    completed_indices = set()

    if output_file.exists():
        old_df = pd.read_csv(output_file)

        needed_score_cols = [
            "TADF_structure_score",
            "stability_score",
            "scaffold_novelty_score",
            "fragment_novelty_score",
            "environment_performance_score",
            "S_LLM"
        ]

        if "source_index" in old_df.columns and all(col in old_df.columns for col in needed_score_cols):
            old_df = old_df.drop_duplicates(subset="source_index", keep="last")
            completed_indices = set(old_df["source_index"].dropna().astype(int).tolist())
            results = old_df.to_dict("records")

            print(f"Existing output file detected: {output_file}")
            print(f"Completed rows: {len(completed_indices)}. Continuing remaining tasks.")
        else:
            print(
                "Existing output file detected, but source_index or required scoring columns "
                "are missing. Recomputing from scratch."
            )
            results = []
            completed_indices = set()

    pending_rows = []

    for _, row in df.iterrows():
        source_index = int(row["source_index"])
        if source_index not in completed_indices:
            pending_rows.append(row)

    total_count = len(df)
    completed_old = len(completed_indices)
    pending_count = len(pending_rows)

    print("----------------------------------------")
    print(f"Total molecules: {total_count}")
    print(f"Already completed: {completed_old}")
    print(f"Pending molecules: {pending_count}")
    print("----------------------------------------")

    if pending_count == 0:
        out_df = pd.DataFrame(results)
        out_df = prepare_output_df(out_df)
        out_df.to_csv(output_file, index=False)

        file_elapsed = time.time() - file_start_time

        print("========================================")
        print(f"No additional scoring needed for file: {input_file}")
        print(f"End time: {current_time_str()}")
        print(f"Total file time: {format_seconds(file_elapsed)}")
        print("========================================")
        return

    run_start_time = time.time()
    completed_new = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {}

        for row in pending_rows:
            source_index = int(row["source_index"])
            future = executor.submit(score_one_molecule, row)
            future_to_index[future] = source_index

        for future in as_completed(future_to_index):
            source_index = future_to_index[future]
            row = df.loc[source_index]

            try:
                score_result = future.result()
            except Exception as e:
                print(f"source_index={source_index} scoring failed: {e}")
                score_result = {
                    "TADF_structure_score": None,
                    "stability_score": None,
                    "scaffold_novelty_score": None,
                    "fragment_novelty_score": None,
                    "environment_performance_score": None,
                    "S_LLM": None,
                    "api_time_seconds": None,
                    "api_time_formatted": None,
                    "reason": "Parallel task failed"
                }

            new_row = row.to_dict()
            new_row.update(score_result)
            results.append(new_row)

            completed_new += 1

            elapsed_this_run = time.time() - run_start_time
            avg_time_per_mol = elapsed_this_run / completed_new
            remaining_new = pending_count - completed_new
            estimated_remaining_time = avg_time_per_mol * remaining_new

            total_finished = completed_old + completed_new
            total_remaining = total_count - total_finished

            print(
                f"Completed in this run {completed_new}/{pending_count} | "
                f"Total progress {total_finished}/{total_count} | "
                f"source_index={source_index} | "
                f"S_LLM={score_result.get('S_LLM')} | "
                f"API time={score_result.get('api_time_seconds')} seconds"
            )

            print(
                f"Current time: {current_time_str()} | "
                f"Elapsed in this run: {format_seconds(elapsed_this_run)} | "
                f"Average time: {format_seconds(avg_time_per_mol)} / molecule | "
                f"Estimated remaining time: {format_seconds(estimated_remaining_time)} | "
                f"Remaining molecules: {total_remaining}"
            )

            print("----------------------------------------")

            if completed_new % save_every == 0:
                temp_df = pd.DataFrame(results)
                temp_df = prepare_output_df(temp_df)
                temp_df.to_csv(output_file, index=False)

                print(f"Temporary results saved to: {output_file}")
                print(f"Save time: {current_time_str()}")
                print("----------------------------------------")

    out_df = pd.DataFrame(results)
    out_df = prepare_output_df(out_df)
    out_df.to_csv(output_file, index=False)

    file_elapsed = time.time() - file_start_time

    print("========================================")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Total molecules: {total_count}")
    print(f"Final scored molecules: {len(out_df)}")
    print(f"End time: {current_time_str()}")
    print(f"Total file time: {format_seconds(file_elapsed)}")
    print("========================================")


# =======================
# 10. Main Program
# =======================

if __name__ == "__main__":
    total_start_time = time.time()

    print("########################################")
    print("All scoring tasks started")
    print(f"Start time: {current_time_str()}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Candidate directory: {CANDIDATE_DIR}")
    print(f"Model name: {MODEL_NAME}")
    print(f"Number of concurrent requests: {MAX_WORKERS}")
    print("########################################")

    for input_file, output_file in INPUT_OUTPUT_PAIRS:
        score_csv_parallel(
            input_file=input_file,
            output_file=output_file,
            save_every=200,
            max_workers=MAX_WORKERS
        )

    total_elapsed = time.time() - total_start_time

    print("########################################")
    print("All files have been scored")
    print(f"End time: {current_time_str()}")
    print(f"Total task time: {format_seconds(total_elapsed)}")
    print("########################################")
