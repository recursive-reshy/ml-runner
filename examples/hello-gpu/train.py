import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

DEFAULT_STEPS = 5
DEFAULT_PAUSE_SECONDS = 2.0
GPU_QUERY = [ "nvidia-smi", "--query-gpu=name", "--format=csv,noheader" ]

# Prints what the GPU check sees; no GPU (a laptop, a CPU job) is a normal answer, not an error.
def detect_gpu() -> str | None:
    if shutil.which( GPU_QUERY[ 0 ] ) is None:
        return None

    completed = subprocess.run( GPU_QUERY, capture_output = True, text = True )
    names = completed.stdout.strip()

    return names if completed.returncode == 0 and names else None

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument( "--steps", type = int, default = DEFAULT_STEPS )
    parser.add_argument( "--pause", type = float, default = DEFAULT_PAUSE_SECONDS )
    arguments = parser.parse_args()

    # Defaults keep `python train.py` working on a laptop with no ml-runner around.
    output_dir = Path( os.environ.get( "OUTPUT_DIR", "out" ) )
    checkpoint_dir = Path( os.environ.get( "CHECKPOINT_DIR", "checkpoints" ) )
    run_id = os.environ.get( "RUN_ID", "local" )
    output_dir.mkdir( parents = True, exist_ok = True )
    checkpoint_dir.mkdir( parents = True, exist_ok = True )

    gpu = detect_gpu()
    print( f"GPU: {gpu or 'none'}", flush = True )

    for step in range( 1, arguments.steps + 1 ):
        time.sleep( arguments.pause )
        ( checkpoint_dir / f"step-{step}.txt" ).write_text( f"step {step}\n" )
        print( f"step {step} of {arguments.steps}", flush = True )

    summary = { "run_id": run_id, "gpu": gpu, "steps": arguments.steps }
    ( output_dir / "result.json" ).write_text( json.dumps( summary, indent = 2 ) )
    print( "done", flush = True )

if __name__ == "__main__":
    main()
