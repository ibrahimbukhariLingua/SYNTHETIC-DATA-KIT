import os
import json
import re
import concurrent.futures
from pathlib import Path
from typing import Optional, Dict, Any, List


from synthetic_data_kit.utils.config import get_prompt


# =================================== Helper Functions ===================================

def parse_thinking_answer(response: str) -> Dict[str, str]:
    thinking_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    thinking = thinking_match.group(1).strip() if thinking_match else ""
    response_without_thinking = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    return {"thinking": thinking, "answer": response_without_thinking}

def parse_json_from_response(response: str) -> Dict[str, Any]:
    triple_backtick_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if triple_backtick_match:
        json_str = triple_backtick_match.group(1)
    else:
        json_match = re.search(r"(\{.*\})", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            raise ValueError("No JSON found in response.")
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

def question_response(question:str, client):
    messages = [
        {"role": "system", "content": """Answer in a short manner and give your final answer within a json as follows {\"answer\":<your answer>}"""},
        {"role": "user", "content": question} 
    ]
    response = client.chat_completion(messages, temperature=0.3)
    try:
        output = parse_output(response)
        return output['answer']
    except Exception as e:
        print("ERROR: ", e)
        return None

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
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise RuntimeError(f"Error parsing output: {e}")

# =================================== Pipeline Sub-Functions ===================================

def clean_and_filter_check(
    qa_pair: dict,
    clean_prompt_template: str,
    filter_prompt_template: str,
    client,
    verbose: bool = False
) -> bool:
    if verbose:
        print("[STAGE] Cleaning QA pair")

    clean_prompt = clean_prompt_template.format(
        chunk=qa_pair['chunk'],
        question=qa_pair['question'],
        answer=qa_pair['answer']
    )
    clean_messages = [
        {"role": "system", "content": "You are a question answers rating system which is provided with an additional passage to enhance and help with the rating."},
        {"role": "user", "content": clean_prompt}
    ]
    try:
        clean_response = client.chat_completion(clean_messages, temperature=0.3)
        rating = int(parse_output(clean_response)['rating'])
    except Exception as e:
        if verbose:
            print(f"[ERROR] Cleaning rating failed: {e}")
        return False

    if rating < 8:
        return False

    if verbose:
        print("[STAGE] Generating synthetic answer")

    synthetic_answer = None
    for _ in range(3):
        try:
            synthetic_answer = question_response(qa_pair['question'], client)
            if synthetic_answer:
                break
        except Exception:
            continue

    if not synthetic_answer:
        return False

    if verbose:
        print("[STAGE] Filtering synthetic vs original answer")

    filter_prompt = filter_prompt_template.format(
        sentence_1=qa_pair['answer'],
        sentence_2=synthetic_answer
    )
    filter_messages = [
        {"role": "system", "content": "You are a sentence similarity rating classifier."},
        {"role": "user", "content": filter_prompt}
    ]
    try:
        filter_response = client.chat_completion(filter_messages, temperature=0.3)
        label = int(parse_output(filter_response)['label'])
        return label == 0
    except Exception as e:
        if verbose:
            print(f"[ERROR] Filtering failed: {e}")
        return False

def reason_cite_single_pair(
    qa_pair: dict,
    asking_prompt: str,
    judge_prompt: str,
    client,
    verbose: bool = False
) -> Optional[dict]:
    if verbose:
        print("[STAGE] Generating reasoned and cited answer")

    system_1 = "You are a Question answering system that uses the supporting context to answer as accurately as possible"
    user_prompt_1 = asking_prompt.format(chunk=qa_pair["chunk"], question=qa_pair["question"])
    messages_1 = [{"role": "system", "content": system_1}, {"role": "user", "content": user_prompt_1}]

    generated_answer, generated_thinking = None, None
    for _ in range(3):
        try:
            response_1 = client.chat_completion(messages_1, temperature=0.3)
            parsed = parse_thinking_answer(response_1)
            generated_answer, generated_thinking = parsed.get("answer", "").strip(), parsed.get("thinking", "").strip()
            if "<statement>" in generated_answer and "</statement>" in generated_answer and "<cite>" in generated_answer and "</cite>" in generated_answer:
                break
        except Exception:
            continue
    else:
        return None

    if verbose:
        print("[STAGE] Comparing generated answer with original")

    system_2 = "You are a sentence comparer which classifies if two sentences match or not"
    user_prompt_2 = judge_prompt.format(
        question=qa_pair["question"],
        gold_answer=qa_pair["answer"],
        our_answer=generated_answer
    )
    messages_2 = [{"role": "system", "content": system_2}, {"role": "user", "content": user_prompt_2}]

    for _ in range(3):
        try:
            response_2 = client.chat_completion(messages_2, temperature=0.3)
            result = parse_json_from_response(response_2)
            if int(result.get("label", "0")) == 1:
                new_pair = dict(qa_pair)
                new_pair["answer"] = generated_answer
                new_pair["thinking"] = generated_thinking
                return new_pair
        except Exception:
            continue
    return None

# =================================== Main Pipeline Function ===================================

def run_pipeline(
    client,
    clean_prompt: str,
    filter_prompt: str,
    reason_prompt: str,
    judge_prompt: str,
    pair: Dict[str, Any],
    verbose: bool = False
) -> Optional[Dict[str, Any]]:
    if verbose:
        print("[PIPELINE] Starting pipeline for question:", pair['question'])

    if not clean_and_filter_check(pair, clean_prompt, filter_prompt, client, verbose):
        return None
    return reason_cite_single_pair(pair, reason_prompt, judge_prompt, client, verbose)

# =================================== Main Processing Function ===================================

def process_file(
    file_path: str,
    client,
    config,
    output_dir: str = "data/preprocess_citations",
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
        print(f"[INIT] Loaded {len(qa_pairs)} QA pairs")

    clean_prompt = get_prompt(config, "qa_rating_w_reasoning")
    filter_prompt = get_prompt(config, "qa_filtering")
    reason_prompt = get_prompt(config, "asking_w_reason")
    judge_prompt = get_prompt(config, "reasoning_judge")

    updated_qa = []

    def batch_worker(batch):
        return [result for pair in batch if (result := run_pipeline(client, clean_prompt, filter_prompt, reason_prompt, judge_prompt, pair, verbose)) is not None]

    batches = [qa_pairs[i:i + batch_size] for i in range(0, len(qa_pairs), batch_size)]

    if verbose:
        print(f"[INFO] Processing {len(batches)} batches with {max_workers} workers")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(batch_worker, batch) for batch in batches]
        for future in concurrent.futures.as_completed(futures):
            updated_qa.extend(future.result())

    output_data = {
        "summary": data.get("summary", ""),
        "qa_pairs": updated_qa,
        "all_chunks": data.get("all_chunks", ""),
    }

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_reasonCited.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    if verbose:
        print(f"\n[COMPLETE] Pipeline complete")
        print(f"- Total pairs processed: {len(qa_pairs)}")
        print(f"- Valid pairs saved: {len(updated_qa)}")
        print(f"- Success rate: {(len(updated_qa)/len(qa_pairs))*100:.1f}%")

    print(f"Saved final QA pairs to {output_path}")
    return output_path
