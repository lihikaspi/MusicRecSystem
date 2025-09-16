# run_all.py
import subprocess
import argparse
import sys
from config import STAGE_FILES

def run_stage(stage_name, stage_file):
    print(f"\n>>> Running stage: {stage_name} ({stage_file}) ...")
    subprocess.run([sys.executable, stage_file], check=True)
    print(f">>> Finished stage: {stage_name}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the project pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        help=(
            "Run a specific stage by name (download, gnn_prep, train_gnn, ann_search) "
            "or by number (1–4). If omitted, all stages run."
        )
    )
    args = parser.parse_args()

    stage_dict = {name: file for name, file in STAGE_FILES}

    if args.stage:
        stage_arg = args.stage.lower()

        if stage_arg.isdigit():
            idx = int(stage_arg) - 1
            if idx < 0 or idx >= len(STAGE_FILES):
                sys.exit(f"❌ Invalid stage number: {stage_arg}")
            stage_name, stage_file = STAGE_FILES[idx]
            run_stage(stage_name, stage_file)

        elif stage_arg in stage_dict:
            run_stage(stage_arg, stage_dict[stage_arg])

        else:
            sys.exit(f"❌ Invalid stage: {args.stage}. "
                     f"Choose from {list(stage_dict.keys())} or 1–{len(STAGE_FILES)}.")

    else:
        for name, file in STAGE_FILES:
            run_stage(name, file)

        print("✅ Pipeline finished successfully!")
