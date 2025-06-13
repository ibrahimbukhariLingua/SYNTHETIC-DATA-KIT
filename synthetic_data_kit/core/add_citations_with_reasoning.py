import os, re
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.config import get_prompt, load_config, get_generation_config


def parse_thinking_answer(response: str) -> Dict[str, str]:
    """
    Parses the response string to extract the content inside <thinking>...</thinking>
    as 'thinking' and the rest of the content as 'answer'.
    """
    thinking_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    thinking = thinking_match.group(1).strip() if thinking_match else ""

    # Remove thinking tag from response
    response_without_thinking = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

    return {
        "thinking": thinking,
        "answer": response_without_thinking,
    }

def parse_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extracts and parses the first JSON object found in the response string.
    Supports both raw JSON and JSON enclosed in triple backticks.
    """
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
        parsed_json = json.loads(json_str)
        return parsed_json
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

def getLLMsResponse(
    inputs: list,
    prompt_template: str,
    client,
    system_prompt: str = 'You are a question answering system that reads the supporting context and answers as correctly as possible',
    parse_mode: str = 'thinking'  # Options: 'thinking', 'json'
) -> Dict[str, Any]:
    """
    Sends a prompt to the LLM client, parses the response according to the parse_mode,
    and prints the outputs.

    Args:
        qa_pair (dict): Contains 'chunk', 'question', and 'answer'.
        prompt_template (str): Template string with placeholders {chunk} and {question}.
        client: The LLM client with chat_completion method.
        system_prompt (str): The system prompt to set the LLM's behavior.
        parse_mode (str): Either 'thinking' or 'json'.

    Returns:
        Dict[str, Any]: Parsed result from the LLM's response.
    """
    
    if parse_mode == 'thinking':
        prompt = prompt_template.format(chunk=inputs[0], question=inputs[1])
    elif parse_mode == 'json':
        prompt = prompt_template.format(question=inputs[0], gold_answer=inputs[1], our_answer=inputs[2])

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        response = client.chat_completion(messages, temperature=0.3)

        if parse_mode == "json":
            parsed_response = parse_json_from_response(response)
        else:
            parsed_response = parse_thinking_answer(response)

        # print(f"Response: {response}")
        # print(f"Parsed: {parsed_response}")

        return parsed_response

    except Exception as e:
        print(f"Error during LLM response or parsing: {e}")
        return {}


def process_file(
    file_path: str,
    config_path: Optional[Path] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    output_dir: str = "data/cited_w_reason",
    batch_size: int = 8,  # Number of QA pairs to process in each batch
    verbose: bool = False
) -> str:
    """
    Process QA pairs using two LLM prompts in batches:
    1. asking_w_reason: generates a cited answer and thinking for all pairs in batch
    2. qa_filtering: checks if the generated and original answers match for all pairs in batch
    
    Uses batch_size to control how many QA pairs are processed in each batch.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    qa_pairs = data.get("qa_pairs", [])
    summary = data.get("summary", "")
    all_chunks = data.get("all_chunks", [])
    if not qa_pairs:
        raise ValueError("No 'qa_pairs' found in input file")

    if verbose:
        print(f"Loaded {len(qa_pairs)} QA pairs")
        print(f"Processing in batches of {batch_size} pairs")

    client = LLMClient(config_path=config_path, api_base=api_base, model_name=model)
    config = client.config

    asking_prompt = get_prompt(config, "asking_w_reason")
    filtering_prompt = get_prompt(config, "reasoning_judge")

    all_cited_pairs = []

    # Process in batches of QA pairs
    for batch_start in range(0, len(qa_pairs), batch_size):
        batch_end = min(batch_start + batch_size, len(qa_pairs))
        current_batch = qa_pairs[batch_start:batch_end]

        if verbose:
            print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(qa_pairs) + batch_size - 1)//batch_size}")
            print(f"Pairs {batch_start+1}-{batch_end} of {len(qa_pairs)}")

        # Step 1: Ask with reason (batch)
        step1_messages = []
        for pair in current_batch:
            messages = [
                {
                    "role": "system",
                    "content": "You are a Question answering system that uses the supporting context to answer as accurately as possible"
                },
                {
                    "role": "user",
                    "content": asking_prompt.format(chunk=pair["chunk"], question=pair["question"])
                }
            ]
            step1_messages.append(messages)

        step1_responses = client.batch_completion(
            message_batches=step1_messages,
            temperature=0.3,
            batch_size=len(step1_messages)
        )

        step1_results = []
        step2_messages = []

        # Retry handling
        for idx, (pair, response) in enumerate(zip(current_batch, step1_responses)):
            success = False
            for attempt in range(3):
                try:
                    parsed_response = parse_thinking_answer(response)
                    generated_answer = parsed_response.get("answer", "").strip()
                    generated_thinking = parsed_response.get("thinking", "").strip()

                    has_statement_tag = "<statement>" in generated_answer and "</statement>" in generated_answer
                    has_citation_tag = "<cite>" in generated_answer and "</cite>" in generated_answer

                    if has_statement_tag and has_citation_tag:
                        step1_results.append({
                            "pair": pair,
                            "generated_answer": generated_answer,
                            "generated_thinking": generated_thinking
                        })
                        step2_messages.append([
                            {
                                "role": "system",
                                "content": "You are a sentence comparer which classifies if two sentences match or not"
                            },
                            {
                                "role": "user",
                                "content": filtering_prompt.format(
                                    question=pair["question"],
                                    gold_answer=pair["answer"],
                                    our_answer=generated_answer
                                )
                            }
                        ])
                        success = True
                        break
                    else:
                        if attempt < 2:
                            if verbose:
                                print(f"[!] Missing tags, retry {attempt+1} for pair {batch_start + idx + 1}")
                            # Retry individually
                            retry_response = client.chat_completion(step1_messages[idx], temperature=0.3)
                            response = retry_response
                        else:
                            if verbose:
                                print(f"[✗] Skipped after 3 retries (missing tags): Pair {batch_start + idx + 1}")
                except Exception as e:
                    if attempt < 2:
                        if verbose:
                            print(f"[!] Retry {attempt+1} due to error in step 1: {e}")
                        retry_response = client.chat_completion(step1_messages[idx], temperature=0.3)
                        response = retry_response
                    else:
                        if verbose:
                            print(f"[✗] Skipped after 3 retries due to parse error: Pair {batch_start + idx + 1}")

        # Step 2: Judge response match
        if step2_messages:
            step2_responses = client.batch_completion(
                message_batches=step2_messages,
                temperature=0.3,
                batch_size=len(step2_messages)
            )

            for idx, (result, response) in enumerate(zip(step1_results, step2_responses)):
                success = False
                for attempt in range(3):
                    try:
                        filter_response = parse_json_from_response(response)
                        label = int(filter_response.get("label", '0'))

                        if label == 1:
                            result_entry = result["pair"]
                            result_entry["answer"] = result["generated_answer"]
                            result_entry["thinking"] = result["generated_thinking"]
                            all_cited_pairs.append(result_entry)
                            if verbose:
                                print(f"[✓] Pair {batch_start + idx + 1} accepted and added")
                        else:
                            if verbose:
                                print(f"[✗] Pair {batch_start + idx + 1} rejected: Label != 1")
                        success = True
                        break
                    except Exception as e:
                        if attempt < 2:
                            if verbose:
                                print(f"[!] Retry {attempt+1} on JSON parse for pair {batch_start + idx + 1}")
                            retry_response = client.chat_completion(step2_messages[idx], temperature=0.3)
                            response = retry_response
                        else:
                            if verbose:
                                print(f"[✗] Skipped after 3 retries in step 2 for pair {batch_start + idx + 1}")

    # Save results
    output_data = {
        'summary': summary,
        'qa_pairs': all_cited_pairs,
        'all_chunks': all_chunks
    }
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_cited_reasoned.jsonl")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    if verbose:
        print(f"\nProcessing complete:")
        print(f"- Total pairs processed: {len(qa_pairs)}")
        print(f"- Valid pairs saved: {len(all_cited_pairs)}")
        print(f"- Success rate: {(len(all_cited_pairs)/len(qa_pairs))*100:.1f}%")
    print(f"Saved Reasoned and Cited output to {output_path}")
    return output_path


