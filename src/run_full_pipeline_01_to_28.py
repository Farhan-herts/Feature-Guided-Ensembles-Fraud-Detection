# -*- coding: utf-8 -*-
"""
Created on Sun May  3 22:47:19 2026

@author: FarhanAli

run_full_pipeline_01_to_28.py

Master runner script to run whole projecct

Purpose:
- Runs project scripts from Step 01 to Step 28 in correct order.
- Stops automatically if any script fails, unless --continue-on-error is used.
- Saves one combined log file in results/master_pipeline_logs/.
- Allows running a partial range using --start and --end.

why I write this script, this script can in one go if we have good specs PC,
like University server.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PIPELINE_STEPS = [
    ("01",  "01_load_merge.py"),
    ("02",  "02_split.py"),
    ("03",  "03_preprocess.py"),
    ("04",  "04_baselines.py"),
    ("05",  "05_feature_views.py"),
    ("06",  "06_bagging_standard.py"),
    ("07",  "07_multiview_bagging.py"),
    ("08",  "08_prepare_feature_views_v2.py"),

    ("09a", "09a_individual_A_bagging.py"),
    ("09b", "09b_individual_B_bagging.py"),
    ("09c", "09c_individual_T_bagging.py"),
    ("09d", "09d_individual_I_bagging.py"),

    ("10",  "10_ab_multiview_bagging.py"),
    ("11",  "11_it_multiview_bagging.py"),
    ("12",  "12_tbi_multiview_bagging.py"),
    ("13",  "13_abi_multiview_bagging.py"),
    ("14",  "14_atb_multiview_bagging.py"),
    ("15",  "15_atbi_multiview_bagging.py"),
    ("16",  "16_compare_all_experiments.py"),

    ("17a", "17a_A_balance_method_comparison.py"),
    ("17b", "17b_B_balance_method_comparison.py"),
    ("17c", "17c_AB_balance_method_comparison.py"),
    ("18",  "18_select_best_balance_method.py"),

    ("19a", "19a_A_best_balanced.py"),
    ("19b", "19b_B_best_balanced.py"),
    ("19c", "19c_T_best_balanced.py"),
    ("19d", "19d_I_best_balanced.py"),

    ("20",  "20_ab_best_balanced.py"),
    ("21",  "21_it_best_balanced.py"),
    ("22",  "22_tbi_best_balanced.py"),
    ("23",  "23_abi_best_balanced.py"),
    ("24",  "24_atb_best_balanced.py"),
    ("25",  "25_atbi_best_balanced.py"),

    ("26",  "26_compare_balanced_vs_unbalanced.py"),
    ("27",  "27_compare_standard_bagging_vs_multiview_ensemble.py"),
    ("28",  "28_exploratory_analysis.py"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all fraud detection project scripts from Step 01 to Step 28."
    )

    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Step key to start from, e.g. 01, 08, 17a, 20."
    )

    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Step key to end at, e.g. 16, 18, 28."
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining scripts even if one script fails."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show which scripts would run, without executing them."
    )

    return parser.parse_args()


def get_step_index(step_key):
    keys = [key for key, _ in PIPELINE_STEPS]
    if step_key not in keys:
        raise ValueError(
            "Unknown step key '{}'. Valid step keys are: {}".format(
                step_key, ", ".join(keys)
            )
        )
    return keys.index(step_key)


def select_steps(start, end):
    start_idx = 0 if start is None else get_step_index(start)
    end_idx = len(PIPELINE_STEPS) - 1 if end is None else get_step_index(end)

    if start_idx > end_idx:
        raise ValueError("--start cannot come after --end in the pipeline order.")

    return PIPELINE_STEPS[start_idx:end_idx + 1]


def make_log_file(project_root):
    log_dir = project_root / "results" / "master_pipeline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / "master_pipeline_{}.log".format(timestamp)


def write_log(log_path, message):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message)


def print_and_log(log_path, message):
    print(message, end="")
    write_log(log_path, message)


def run_one_script(script_path, log_path):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    command = [sys.executable, str(script_path)]

    process = subprocess.Popen(
        command,
        cwd=str(script_path.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1
    )

    if process.stdout is None:
        raise RuntimeError("Could not read script output.")

    for line in process.stdout:
        print_and_log(log_path, line)

    process.wait()
    return int(process.returncode)


def check_required_scripts(script_dir, steps):
    missing = []
    for _, filename in steps:
        path = script_dir / filename
        if not path.exists():
            missing.append(path)
    return missing


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    selected_steps = select_steps(args.start, args.end)
    log_path = make_log_file(project_root)

    header = (
        "\n"
        "============================================================\n"
        " FRAUD DETECTION MASTER PIPELINE: STEP 01 TO 28\n"
        "============================================================\n"
        "Started at     : {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) +
        "Python         : {}\n".format(sys.executable) +
        "Scripts folder : {}\n".format(script_dir) +
        "Project root   : {}\n".format(project_root) +
        "Log file       : {}\n".format(log_path) +
        "Continue error : {}\n".format(args.continue_on_error) +
        "Dry run        : {}\n".format(args.dry_run) +
        "============================================================\n\n"
    )

    print_and_log(log_path, header)

    print_and_log(log_path, "Selected steps:\n")
    for step_key, filename in selected_steps:
        print_and_log(log_path, "  {:>3}  {}\n".format(step_key, filename))
    print_and_log(log_path, "\n")

    missing_scripts = check_required_scripts(script_dir, selected_steps)
    if missing_scripts:
        print_and_log(log_path, "[ERROR] Some required scripts are missing:\n")
        for path in missing_scripts:
            print_and_log(log_path, "  - {}\n".format(path))

        print_and_log(
            log_path,
            "\nFix: Place this master runner inside the same scripts folder as your Step 01-28 scripts.\n"
        )
        sys.exit(1)

    if args.dry_run:
        print_and_log(log_path, "[DRY RUN] No scripts executed.\n")
        return

    summary_rows = []
    pipeline_start = time.time()

    for step_number, step in enumerate(selected_steps, start=1):
        step_key, filename = step
        script_path = script_dir / filename

        section_header = (
            "\n"
            "------------------------------------------------------------\n"
            "Running {}: {}\n".format(step_key, filename) +
            "Step {} of {}\n".format(step_number, len(selected_steps)) +
            "Started: {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) +
            "------------------------------------------------------------\n"
        )
        print_and_log(log_path, section_header)

        start_time = time.time()
        return_code = run_one_script(script_path, log_path)
        elapsed = time.time() - start_time

        status = "SUCCESS" if return_code == 0 else "FAILED"

        summary_rows.append({
            "step": step_key,
            "script": filename,
            "status": status,
            "return_code": return_code,
            "seconds": round(elapsed, 2)
        })

        result_msg = (
            "\n"
            "[{}] {}: {}\n".format(status, step_key, filename) +
            "Return code: {}\n".format(return_code) +
            "Time taken : {:.2f} seconds\n".format(elapsed)
        )
        print_and_log(log_path, result_msg)

        if return_code != 0 and not args.continue_on_error:
            print_and_log(
                log_path,
                "\nPipeline stopped because a script failed.\n"
                "Use --continue-on-error only if you intentionally want to continue after failures.\n"
            )
            break

    total_elapsed = time.time() - pipeline_start

    print_and_log(
        log_path,
        "\n============================================================\n"
        " PIPELINE SUMMARY\n"
        "============================================================\n"
    )

    for row in summary_rows:
        print_and_log(
            log_path,
            "{:>3} | {:<7} | {:>10.2f}s | {}\n".format(
                row["step"], row["status"], row["seconds"], row["script"]
            )
        )

    failed = [row for row in summary_rows if row["status"] != "SUCCESS"]

    print_and_log(log_path, "\n")
    print_and_log(log_path, "Total time: {:.2f} seconds\n".format(total_elapsed))
    print_and_log(log_path, "Log file  : {}\n".format(log_path))

    if failed:
        print_and_log(log_path, "\nSome steps failed:\n")
        for row in failed:
            print_and_log(
                log_path,
                "  - {} {} returned code {}\n".format(
                    row["step"], row["script"], row["return_code"]
                )
            )
        sys.exit(1)

    print_and_log(log_path, "\nAll selected pipeline steps completed successfully.\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[MASTER PIPELINE ERROR] {}: {}".format(type(exc).__name__, exc))
        sys.exit(1)
