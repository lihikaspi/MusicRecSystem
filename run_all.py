import subprocess
import argparse
import sys
from config import config

def run_stage(stage_name: str, stage_file: str):
    print(f"\n>>> Running stage: {stage_name} ({stage_file}) ...")
    subprocess.run([sys.executable, stage_file], check=True)
    print(f">>> Finished stage: {stage_name}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the project pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        help=(
            "Run a specific stage by name (download, gnn_prep, gnn_train, ann_search) "
            "or by number (1–4). If omitted, all stages run."
        )
    )
    args = parser.parse_args()

    stage_files = config.pipeline.stage_files
    stage_dict = {name: file for name, file in stage_files}

    if args.stage:
        stage_arg = args.stage.lower()

        if stage_arg.isdigit():
            idx = int(stage_arg) - 1
            if idx < 0 or idx >= len(stage_files):
                sys.exit(f"Invalid stage number: {stage_arg}")
            stage_name, stage_file = stage_files[idx]
            run_stage(stage_name, stage_file)

        elif stage_arg in stage_dict:
            run_stage(stage_arg, stage_dict[stage_arg])

        else:
            sys.exit(f"Invalid stage: {args.stage}. "
                     f"Choose from {list(stage_dict.keys())} or 1–{len(stage_files)}.")

    else:
        for name, file in stage_files:
            run_stage(name, file)

    print(">>> Pipeline finished successfully!")
