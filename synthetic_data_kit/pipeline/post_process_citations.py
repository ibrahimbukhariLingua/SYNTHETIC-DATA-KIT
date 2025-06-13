import os
import json
import re
import concurrent.futures
from pathlib import Path
from typing import Optional, Dict, Any, List

from synthetic_data_kit.utils.config import get_prompt

def parse_output(output: str) -> Dict[str, Any]:
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        pass

    try:
        match = re.search(r'\{.*?\}', output, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in the output.")

        json_str = match.group(0)
        parsed_output = json.loads(json_str)
        return parsed_output
    except json.JSONDecodeError as e:
        return None
    except Exception as e:
        return None

# =================================== Pipeline Functions ===================================

def filter_citation(client, prompt_template: str, chunk: str, question: str, answer: str) -> Dict[str, Any]:
    prompt = prompt_template.format(chunk=chunk, question=question, answer=answer)
    messages = [
        {"role": "system", "content": "You are a professional QA evaluator who checks if citations properly support answers."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat_completion(messages=messages, temperature=0.3)
    return parse_output(response)

def combine_citation(client, prompt_template: str, answer: str) -> Dict[str, Any]:
    prompt = prompt_template.format(answer=answer)
    messages = [
        {"role": "system", "content": "You are an expert assistant that reformulates QA pairs with proper citations."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat_completion(messages=messages, temperature=0.3)
    return parse_output(response)

def run_pipeline(client, filter_prompt: str, combine_prompt: str, pair: Dict[str, Any], verbose: bool = False, batch_idx: int = 0, pipeline_idx: int = 0) -> Optional[Dict[str, Any]]:
    chunk = pair.get("chunk", "")
    question = pair["question"]
    answer = pair["answer"]

    if verbose:
        print(f"[DEBUG][BATCH {batch_idx}][PIPELINE {pipeline_idx}] Starting pipeline ...")

    rating_data = None
    for attempt in range(3):
        rating_data = filter_citation(client, filter_prompt, chunk, question, answer)
        if verbose:
            print(f"[DEBUG][BATCH {batch_idx}][PIPELINE {pipeline_idx}] filter_citation attempt {attempt+1}")
        if rating_data is not None:
            break
    if rating_data is None or rating_data.get("rating", 0) < 8:
        if verbose:
            print(f"[DEBUG][BATCH {batch_idx}][PIPELINE {pipeline_idx}] Filter rejected or rating too low.")
        return None

    combined = None
    for attempt in range(3):
        combined = combine_citation(client, combine_prompt, answer)
        if verbose:
            print(f"[DEBUG][BATCH {batch_idx}][PIPELINE {pipeline_idx}] combine_citation attempt {attempt+1}")
        if combined is not None:
            break
    if combined is None:
        if verbose:
            print(f"[DEBUG][BATCH {batch_idx}][PIPELINE {pipeline_idx}] Combine failed.")
        return None

    new_answer = combined.get("answer", "")
    if not new_answer or not new_answer.strip().startswith("<statement>"):
        if verbose:
            print(f"[DEBUG][BATCH {batch_idx}][PIPELINE {pipeline_idx}] New answer is invalid.")
        return None

    pair["answer"] = new_answer
    return pair

# =================================== Main Processing Function ===================================

def process_file(
    file_path: str,
    client, 
    config,
    output_dir: str = "data/postprocess_citations",
    verbose: bool = False,
    max_workers: int = 4,
    batch_size: int = 8
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    qa_pairs = data.get("qa_pairs", [])
    if not qa_pairs:
        raise ValueError("No 'qa_pairs' found in input file")

    if verbose:
        print(f"[DEBUG] Loaded {len(qa_pairs)} QA pairs")

    filter_prompt = get_prompt(config, "filter_citations")
    combine_prompt = get_prompt(config, "combine_citations")

    updated_qa = []

    def batch_worker(batch, batch_idx):
        results = []
        for pipeline_idx, pair in enumerate(batch):
            result = run_pipeline(client, filter_prompt, combine_prompt, pair, verbose=verbose, batch_idx=batch_idx, pipeline_idx=pipeline_idx)
            if result is not None:
                results.append(result)
        return results

    batches = [qa_pairs[i:i + batch_size] for i in range(0, len(qa_pairs), batch_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(batch_worker, batch, idx) for idx, batch in enumerate(batches)]
        for future in concurrent.futures.as_completed(futures):
            updated_qa.extend(future.result())

    output_data = {
        "summary": data.get("summary", ""),
        "qa_pairs": updated_qa,
        "all_chunks": data.get("all_chunks", ""),
    }

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_final.json")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    if verbose:
        print(f"\n[DEBUG] Final pipeline complete:")
        print(f"[DEBUG] - Total pairs processed: {len(qa_pairs)}")
        print(f"[DEBUG] - Valid pairs saved: {len(updated_qa)}")
        print(f"[DEBUG] - Success rate: {(len(updated_qa)/len(qa_pairs))*100:.1f}%")

    print(f"Saved final QA pairs to {output_path}")
    return output_path
