# Basic imports
import os, re
from typing import List
import torch

# HuggingFace imports
from huggingface_hub import list_repo_files
from huggingface_hub.hf_api import HfFolder

# Local imports
from synthetic_data_kit.evaluation.finetune import Finetune_w_checkpoint
# from synthetic_data_kit.evaluation.evaluation_v2 import Finetune
from synthetic_data_kit.evaluation.evaluation import ModelEvaluator
from synthetic_data_kit.evaluation.util import generate_ft_model_name

# Set the environment variable for HuggingFace token
HfFolder.save_token("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")


def run_finetuning_all_configs(
    rag_index_dir: str,
    dataset_dirs: List[str],
    model_variants: List[str],
    version: str,
    dataset_size:int = 1000,
    verbose: bool = False,
):
    hf_user = "ibrahimbukhariLingua"

    def log_debug(message: str):
        if verbose:
            print(f"[DEBUG] {message}")

    def is_model_already_uploaded(ft_model_name: str) -> bool:
        repo_id = f"{hf_user}/{ft_model_name}"
        log_debug(f"Checking HuggingFace for existing model: {repo_id}")
        try:
            files = list_repo_files(repo_id)
            exists = bool(files)
            log_debug(f"Model found: {exists}")
            return exists
        except Exception as e:
            log_debug(f"Exception while checking HuggingFace repo: {e}")
            return False

    for dataset_dir in dataset_dirs:
        log_debug(f"Using dataset directory: {dataset_dir}")
        for model in model_variants:
            log_debug(f"Using model variant: {model}")
            log_debug(f"Using dataset size: {dataset_size}")

            ft_model_name = generate_ft_model_name(model, dataset_dir, dataset_size, version)
            log_debug(f"Generated fine-tuned model name: {ft_model_name}")

            if is_model_already_uploaded(ft_model_name):
                print(f"[SKIP] {ft_model_name} already exists on HuggingFace.")
                continue

            kwargs = {
                "rag_index_dir": rag_index_dir,
                "input_dir": dataset_dir,
                "model_name": model,
                "device_map": "cuda",
                "num_of_samples": dataset_size,
                "ft_model_name": ft_model_name,
            }

            print(f"[RUNNING] Fine-tuning {ft_model_name} with args: {kwargs}")
            try:
                torch.cuda.empty_cache()
                log_debug("Cleared CUDA memory.")
                # finetuning = Finetune(**kwargs)
                finetuning = Finetune_w_checkpoint(**kwargs)
                log_debug("Finetune instance created.")
                finetuning.run()
                print(f"[SUCCESS] Finished fine-tuning {ft_model_name}")
            except Exception as e:
                print(f"[ERROR] Failed to fine-tune {ft_model_name}: {e}")
                break  # Stop further attempts if one fails


def run_evaluation_from_models_folder(
    benchmarks: List[str],
    version: str,
    models_base_dir:str = "Models",
    verbose: bool = False
):
    all_results = []

    valid_checkpoints = {"checkpoint-62", "checkpoint-125"}

    def get_matching_model_dirs(base_dir: str, version: str):
        matched_dirs = []
        dataset = ""
        pattern = re.compile(rf".*{dataset}-{re.escape(version)}$")
        for root, dirs, _ in os.walk(base_dir):
            for d in dirs:
                if pattern.match(d):
                    matched_dirs.append(os.path.join(root, d))
        return matched_dirs

    def get_target_checkpoints(model_dir: str):
        return [
            os.path.join(model_dir, ckpt)
            for ckpt in os.listdir(model_dir)
            if ckpt in valid_checkpoints and os.path.isdir(os.path.join(model_dir, ckpt))
        ]

    model_dirs = get_matching_model_dirs(models_base_dir, version)

    for model_dir in model_dirs:
        checkpoints = get_target_checkpoints(model_dir)
        for checkpoint_path in checkpoints:
            try:
                
                # checks if there is a directory with the same name as result_check_name in the location data/results
                result_check_name = checkpoint_path.replace("/", "_")
                result_check_dir = os.path.join("data", "results", result_check_name)
                if os.path.exists(result_check_dir):
                    if verbose:
                        print(f"[SKIP] {result_check_name} has already been evaluated: {result_check_dir}")
                    continue
                
                # Initialize the evaluator with the checkpoint path
                if verbose:
                    print(f"[INFO] Evaluating checkpoint: {checkpoint_path}")
                evaluator = ModelEvaluator(model_name=checkpoint_path, verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"[ERROR] Failed to initialize evaluator for {checkpoint_path}: {e}")
                continue

            # Evaluate the model on each benchmark
            checkpoint_results = {checkpoint_path: []}
            for benchmark in benchmarks:
                try:
                    result = evaluator.evaluate_dataset(benchmark, "test")
                    if verbose:
                        print(f"[DEBUG] Evaluation result for {benchmark}: {result}")
                    checkpoint_results[checkpoint_path].append(result)
                except Exception as e:
                    if verbose:
                        print(f"[ERROR] Failed to evaluate benchmark {benchmark} for {checkpoint_path}: {e}")
                    continue

            all_results.extend(checkpoint_results.items())

    return all_results

