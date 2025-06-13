import os, time
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

from synthetic_data_kit.pipeline.pre_process_citations import process_file as preprocess
from synthetic_data_kit.pipeline.post_process_citations import process_file as postprocess

# =================================== Main Pipeline Function ===================================

def pipeline(file_path, output_dir, client, config, generator, num_pairs: int, verbose:bool):
    """
    Run the pipeline for a single file.
    
    Args:
        file_path: Path to input file
        output_dir: Directory to save output
        config_path: Path to config file
        api_base: API base URL
        model: Model name
        num_pairs: Number of QA pairs to generate
        
    Returns:
        tuple: (Path to final output file, Path to buffer directory)
    """
    file_path = Path(file_path)
    input_dir_name = file_path.parent.name
    output_dir = Path(output_dir)

    if input_dir_name == "output":
        date_str = datetime.now().strftime("%Y-%m-%d")
        new_output_dir = next((output_dir / f"{date_str}_v{i}_pipeline_v2" for i in range(1, 1000) if not (output_dir / f"{date_str}_v{i}_pipeline_v2").exists()), None)
    else:
        new_output_dir = output_dir / f"{input_dir_name}_pipeline_v2"

    new_output_dir.mkdir(parents=True, exist_ok=True)
    buffer_dir = new_output_dir / "buffer"
    buffer_dir.mkdir(exist_ok=True)

    try:
        qa_path = generator.process_document(
            input_file_path=str(file_path),
            num_pairs=num_pairs,
            output_dir=str(buffer_dir),
            verbose=verbose
        )
        time.sleep(1)

        pre_path = preprocess(
            file_path=qa_path,
            output_dir=buffer_dir,
            client=client,
            config=config,
            verbose=verbose
        )
        time.sleep(1)

        final_path = postprocess(
            file_path=pre_path,
            output_dir=new_output_dir,
            client=client,
            config=config,
            verbose=verbose
        )
        time.sleep(1)

        return final_path, buffer_dir

    except Exception as e:
        raise RuntimeError(f"Pipeline failed for {file_path}: {e}")


# =================================== Single Sample ===================================

def process_single_file(file_path: Path, output_dir: Path, client, config, generator, num_pairs: int, verbose:bool) -> Tuple[int, int, Path]:
    try:
        output_json_path, buffer_dir = pipeline(file_path, output_dir, client, config, generator, num_pairs, verbose)

        with open(output_json_path, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f).get("qa_pairs", [])

        if not qa_pairs:
            Path(output_json_path).unlink()
            return 0, 0, buffer_dir
        else:
            return len(qa_pairs), 1, buffer_dir

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, 0, None



# =================================== Main Processing Function ===================================

def process_file(input_dir, output_dir, num_files, client, config, generator, num_pairs: int, verbose:bool, max_workers: Optional[int] = None):
    input_dir = Path(input_dir)
    log_dir = input_dir / "log_v2"
    unprocessed_file = log_dir / "unprocessed_texts.txt"

    if unprocessed_file.exists():
        with unprocessed_file.open('r', encoding='utf-8') as f:
            unprocessed = [Path(line.strip()) for line in f if line.strip()]
    else:
        unprocessed = list(input_dir.rglob("*.txt"))

    if not unprocessed:
        print("No unprocessed files found.")
        return

    to_process = unprocessed[:num_files]
    log_dir.mkdir(exist_ok=True)

    if max_workers is None:
        max_workers = max(1, cpu_count() - 1)
    max_workers = min(max_workers, len(to_process))

    print(f"Processing {len(to_process)} files using {max_workers} workers...")

    process_func = partial(
        process_single_file,
        output_dir=Path(output_dir),
        client=client,
        config=config,
        generator=generator,
        num_pairs=num_pairs,
        verbose=verbose
    )

    buffer_dirs = set()
    try:
        with Pool(max_workers) as pool:
            results = pool.map(process_func, to_process)

        total_qa_pairs = 0
        processed_files = 0
        for pairs, success, buffer_dir in results:
            total_qa_pairs += pairs
            processed_files += success
            if buffer_dir is not None:
                buffer_dirs.add(buffer_dir)

        if processed_files > 0:
            remaining = [str(f) for f in unprocessed[processed_files:]]
            unprocessed_file.write_text('\n'.join(remaining), encoding='utf-8')

        print(f"Processed files: {processed_files}")
        print(f"Total QA pairs generated: {total_qa_pairs}")

    finally:
        print("Cleaning up buffer directories...")
        for buffer_dir in buffer_dirs:
            try:
                if buffer_dir.exists():
                    shutil.rmtree(buffer_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up buffer directory {buffer_dir}: {e}")
