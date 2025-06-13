import os
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.config import get_prompt


def parse_output(output: str) -> Dict[str, Any]:
    """
    Extracts and parses a JSON-like structure from the given output string.

    Args:
        output (str): The string containing the JSON object.

    Returns:
        Dict[str, Any]: Parsed dictionary containing the citation output.
    """
    try:
        # First, try parsing the entire string directly as JSON
        return json.loads(output)
    except json.JSONDecodeError:
        pass  # Fall back to extracting a JSON-like object

    try:
        # Extract JSON object from string using regex
        match = re.search(r'\{.*?\}', output, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in the output.")
        
        json_str = match.group(0)
        parsed_output = json.loads(json_str)
        return parsed_output
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise RuntimeError(f"Error parsing output: {e}")


def prepare_batch_messages(qa_pairs: List[Dict], prompt_template: str, batch_size: int) -> List[List[Dict]]:
    """
    Prepare batches of messages for the LLM.
    
    Args:
        qa_pairs: List of QA pairs to process
        prompt_template: Template for the prompt
        batch_size: Number of pairs to process in each batch
        
    Returns:
        List of batches, where each batch contains message lists for the LLM
    """
    all_batches = []
    current_batch = []
    
    for pair in qa_pairs:
        prompt = prompt_template.format(
            chunk=pair['chunk'],
            question=pair['question'],
            answer=pair['answer']
        )
        
        messages = [
            {
                "role": "system",
                "content": 'You are a professional QA evaluator who checks if citations properly support answers'
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        current_batch.append(messages)
        
        if len(current_batch) >= batch_size:
            all_batches.append(current_batch)
            current_batch = []
    
    # Add any remaining pairs
    if current_batch:
        all_batches.append(current_batch)
    
    return all_batches


def process_file(
    file_path: str,
    config_path: Optional[Path] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    output_dir: str = "data/reasoning_distil",
    verbose: bool = False,
    max_retries: int = 3,
    batch_size: int = 8
) -> str:
    """
    Filter QA pairs based on citation validity.
    
    Args:
        file_path: Path to JSON file with cited QA pairs
        config_path: Path to configuration YAML/JSON
        api_base: Optional LLM API base URL
        model: Model name
        output_dir: Where to save filtered output
        verbose: Print progress
        max_retries: Number of retries for failed API calls
        batch_size: Number of QA pairs to process in each batch
        
    Returns:
        Path to output file with filtered QA pairs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    qa_pairs = data.get("qa_pairs", [])
    if not qa_pairs:
        raise ValueError("No 'qa_pairs' found in input file")

    if verbose:
        print(f"Loaded {len(qa_pairs)} QA pairs")

    # Initialize client and config
    client = LLMClient(config_path=config_path, api_base=api_base, model_name=model)
    config = client.config
    prompt_template = get_prompt(config, "filter_citations")

    filtered_qa = []
    
    # Prepare batches of messages
    batches = prepare_batch_messages(qa_pairs, prompt_template, batch_size)
    
    if verbose:
        print(f"Processing {len(batches)} batches of up to {batch_size} pairs each")
    
    # Process each batch
    for batch_idx, batch in enumerate(batches):
        for attempt in range(max_retries):
            try:
                # Process entire batch at once
                responses = client.batch_completion(
                    message_batches=batch,
                    temperature=0.3,
                    batch_size=len(batch)
                )
                
                # Process responses and add high-quality pairs to filtered list
                start_idx = batch_idx * batch_size
                for i, response in enumerate(responses):
                    try:
                        rating_data = parse_output(response)
                        pair_idx = start_idx + i
                        
                        if pair_idx < len(qa_pairs):  # Ensure we don't go out of bounds
                            # Only keep pairs with rating >= 8 (high quality citations)
                            if rating_data.get('rating', 0) >= 8:
                                filtered_qa.append(qa_pairs[pair_idx])
                            
                            if verbose:
                                print(f"Processed pair {pair_idx + 1} - Rating: {rating_data.get('rating', 0)}")
                    except Exception as e:
                        print(f"Error processing response for pair {start_idx + i + 1}: {e}")
                        continue
                
                break  # Exit retry loop if successful
                
            except Exception as e:
                if attempt < max_retries - 1:
                    if verbose:
                        print(f"Retry {attempt + 1} for batch {batch_idx + 1} due to error: {str(e)}")
                else:
                    print(f"Error in batch {batch_idx + 1} after {max_retries} retries: {str(e)}")

    # Compose final output
    output_data = {
        "summary": data.get("summary", ""),
        "qa_pairs": filtered_qa,
        "all_chunks": data.get("all_chunks", ""),
    }

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_filtered.json")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    if verbose:
        print(f"\nFiltering complete:")
        print(f"- Total pairs processed: {len(qa_pairs)}")
        print(f"- Valid pairs saved: {len(filtered_qa)}")
        print(f"- Success rate: {(len(filtered_qa)/len(qa_pairs))*100:.1f}%")
    
    print(f"Saved filtered QA pairs to {output_path}")
    return output_path 