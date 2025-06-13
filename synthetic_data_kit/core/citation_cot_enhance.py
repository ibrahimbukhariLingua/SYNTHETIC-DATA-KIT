import os, re
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.config import get_prompt, load_config, get_generation_config



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




def process_file(
    file_path: str,
    config_path: Optional[Path] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    output_dir: str = "data/cot_cited",
    verbose: bool = False,
    max_retries = 3
) -> str:
    """
    Add citations to QA pairs using an LLM.
    
    Args:
        file_path: Path to JSON file with `qa_pairs` list
        config_path: Path to configuration YAML/JSON
        api_base: Optional LLM API base URL
        model: Model name
        output_dir: Where to save Reasoning-enhanced output
        batch_size: Number of pairs per LLM batch
        verbose: Print progress
        
    Returns:
        Path to output file with citations
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
    prompt_template = get_prompt(config, "cot_citation_enhancement")

    all_cot_qa = []
    
    for i, pair in enumerate(qa_pairs):
        
        prompt = prompt_template.format(chunk = pair['chunk'], question=pair['question'], answer=pair['answer'])
        messages = [{"role": "system", "content": 'You are a professional Chain-of-Thought Reasoning adder who follows the instructions provided and adds reasoning to the text question answer pair'},
                    {"role": "user", "content": prompt}]

        for attempt in range(max_retries):
            try:
                response = client.chat_completion(messages, temperature=0.3)
                
                cot_pair = parse_output(response)
                
                cot_pair['chunk'] = pair["chunk"]
                cot_pair['question'] = pair["question"]
                cot_pair['answer'] = pair["answer"]
                # cot_pair['rating'] = pair["rating"]
                
                
                if cot_pair.get('reasoning', "NONE") != "NONE":
                    all_cot_qa.append(cot_pair)
                
                if verbose:
                    print(f"Processed pair {i + 1}")
                break  # Exit the retry loop if successful

            except Exception as e:
                if attempt < max_retries - 1:
                    if verbose:
                        print(f"Retry {attempt + 1} for pair {i + 1} due to error: {str(e)}")
                else:
        
                    print(f"Error in Pair {i + 1} after {max_retries} retries: {str(e)}")
                    print(response)

    # Compose final output
    output_data = {
        "summary":data.get("summary", []),
        "qa_pairs": all_cot_qa
    }

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_cotEnhanced.json")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved Reasoning QA pairs with citations to {output_path}")
    return output_path
