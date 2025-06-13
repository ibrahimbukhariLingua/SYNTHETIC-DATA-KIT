import os, time
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

from synthetic_data_kit.core.create import process_file as create_qa
from synthetic_data_kit.core.curate_w_reason import process_file as clean_w_reason
from synthetic_data_kit.core.add_citations_with_reasoning import process_file as add_reasoning
from synthetic_data_kit.core.filter_citations import process_file as filter_citations


def process_single_file(file_path: Path, output_dir: Path, config_path: Path, api_base: str, model: str) -> Tuple[int, int, Path]:
    """
    Process a single file through the pipeline.
    
    Args:
        file_path: Path to the input file
        output_dir: Directory to save output
        config_path: Path to config file
        api_base: API base URL
        model: Model name
        
    Returns:
        tuple: (number of QA pairs, success flag, buffer directory path)
    """
    try:
        # Run the pipeline on the file
        
        # =================== To Be Modified ===================
        output_json_path, buffer_dir = pipeline(file_path, output_dir, config_path, api_base, model)

        # Check QA pairs
        with open(output_json_path, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f).get("qa_pairs", [])

        # Return results without deleting buffer dir
        if not qa_pairs:
            Path(output_json_path).unlink()
            return 0, 0, buffer_dir
        else:
            return len(qa_pairs), 1, buffer_dir

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, 0, None


def process_file(input_dir, output_dir, num_files, config_path, api_base, model, max_workers: Optional[int] = None): 
    """
    Process multiple files in parallel through the pipeline.
    
    Args:
        input_dir: Directory containing input files
        output_dir: Directory to save output
        num_files: Number of files to process
        config_path: Path to config file
        api_base: API base URL
        model: Model name
        max_workers: Maximum number of parallel workers (defaults to CPU count - 1)
    """
    input_dir = Path(input_dir)
    log_dir = input_dir / "log"
    unprocessed_file = log_dir / "unprocessed_texts.txt"

    # Step 1: Load existing unprocessed files list if available, otherwise scan for .txt files
    if unprocessed_file.exists():
        with unprocessed_file.open('r', encoding='utf-8') as f:
            unprocessed = [Path(line.strip()) for line in f if line.strip()]
    else:
        unprocessed = list(input_dir.rglob("*.txt"))

    if not unprocessed:
        print("No unprocessed files found.")
        return

    # Step 2: Select top `num_files` for processing
    to_process = unprocessed[:num_files]

    # Ensure log directory exists
    log_dir.mkdir(exist_ok=True)

    # Determine number of workers
    if max_workers is None:
        max_workers = max(1, cpu_count() - 1)  # Leave one CPU free
    max_workers = min(max_workers, len(to_process))  # Don't use more workers than files

    print(f"Processing {len(to_process)} files using {max_workers} workers...")

    # Create partial function with fixed arguments
    process_func = partial(
        process_single_file,
        output_dir=Path(output_dir),
        config_path=config_path,
        api_base=api_base,
        model=model
    )

    # Process files in parallel
    buffer_dirs = set()
    try:
        with Pool(max_workers) as pool:
            results = pool.map(process_func, to_process)

        # Collect buffer directories and aggregate results
        total_qa_pairs = 0
        processed_files = 0
        for pairs, success, buffer_dir in results:
            total_qa_pairs += pairs
            processed_files += success
            if buffer_dir is not None:
                buffer_dirs.add(buffer_dir)

        # Update unprocessed list
        if processed_files > 0:
            remaining = [str(f) for f in unprocessed[processed_files:]]
            unprocessed_file.write_text('\n'.join(remaining), encoding='utf-8')

        # Print stats
        print(f"Processed files: {processed_files}")
        print(f"Total QA pairs generated: {total_qa_pairs}")

    finally:
        # Clean up all buffer directories after all processes complete
        print("Cleaning up buffer directories...")
        for buffer_dir in buffer_dirs:
            try:
                if buffer_dir.exists():
                    shutil.rmtree(buffer_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up buffer directory {buffer_dir}: {e}")


def pipeline(file_path, output_dir, config_path, api_base, model):
    """
    Run the pipeline for a single file.
    
    Args:
        file_path: Path to input file
        output_dir: Directory to save output
        config_path: Path to config file
        api_base: API base URL
        model: Model name
        
    Returns:
        tuple: (Path to final output file, Path to buffer directory)
    """
    file_path = Path(file_path)
    input_dir_name = file_path.parent.name
    output_dir = Path(output_dir)

    # Step 1: Determine output directory
    if input_dir_name == "output":
        date_str = datetime.now().strftime("%Y-%m-%d")
        new_output_dir = next((output_dir / f"{date_str}_v{i}_reasoning_distilled" for i in range(1, 1000) if not (output_dir / f"{date_str}_v{i}_reasoning_distilled").exists()), None)
    else:
        new_output_dir = output_dir / f"{input_dir_name}_reasoning_distilled"

    # Step 2: Create output/buffer dirs
    new_output_dir.mkdir(parents=True, exist_ok=True)
    buffer_dir = new_output_dir / "buffer"
    buffer_dir.mkdir(exist_ok=True)

    # Step 3: Set shared params
    kwargs = dict(api_base=api_base, model=model, config_path=config_path)

    try:
        # Step 4: Run pipeline stages
        
        qa_path = create_qa(file_path=file_path, output_dir=str(buffer_dir), **kwargs)
        time.sleep(2)
        
        clean_qa_path = clean_w_reason(file_path=qa_path, output_dir=str(buffer_dir), action="clean_only", threshold=8, **kwargs)
        time.sleep(2)
        
        # reason distilled
        cited_qa_path = add_reasoning(file_path=clean_qa_path, output_dir=str(buffer_dir), **kwargs)
        time.sleep(2)
        
        final_path = filter_citations(file_path=cited_qa_path, output_dir=str(new_output_dir), **kwargs)
        time.sleep(2)
        
        return final_path, buffer_dir
        
    except Exception as e:
        raise RuntimeError(f"Pipeline failed for {file_path}: {e}")
