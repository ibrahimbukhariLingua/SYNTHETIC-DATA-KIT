import os, re, time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.config import get_prompt, get_curate_config



# ============================  UTIL Functions =============================== #

# Loading and Initializing Functions
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

def load_qa_data(file_path: str) -> tuple[list[dict], str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    qa_pairs = data.get("qa_pairs", [])
    summary = data.get("summary", "")
    all_chunks = data.get("all_chunks", "")
    if not qa_pairs:
        raise ValueError("No 'qa_pairs' found in input file")
    return qa_pairs, summary, all_chunks

def initialize_client(config_path, api_base, model) -> tuple[LLMClient, str, dict]:
    client = LLMClient(config_path=config_path, api_base=api_base, model_name=model)
    cleaning_prompt_template = get_prompt(client.config, "qa_rating_w_reasoning")
    filering_prompt_template = get_prompt(client.config, "qa_filtering")
    curate_config = get_curate_config(client.config)
    return client, cleaning_prompt_template, filering_prompt_template, curate_config



# Cleaning and Filtering Functions
def qa_cleaning(pair: dict, prompt_template: str, client: LLMClient, max_retries: int, verbose: bool) -> Optional[Dict]:
    prompt = prompt_template.format(chunk=pair['chunk'], question=pair['question'], answer=pair['answer'])
    messages = [
        {"role": "system", "content": "You are a question answers rating system which is provided with an additional passage to enhance and help with the rating."},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(max_retries):
        try:
            time.sleep(1)
            response = client.chat_completion(messages, temperature=0.3)
            rating = parse_output(response)
            return int(rating['rating'])
        except Exception as e:
            if verbose:
                print(f"Retry {attempt + 1} for pair due to error: {e}")
    return None

def question_response(question:str, client: LLMClient):
    messages = [
                {"role": "system", "content": """Answer in a short manner and give your final answer within a json as follows {"answer":<your answer>}"""},
                {"role": "user", "content": question} 
            ]
    response = client.chat_completion(messages, temperature=0.3)
    try:
        output = parse_output(response)
        return output['answer']
    except Exception as e:
        print("ERROR: ", e)
        return None

def qa_filtering(sentence1:str, sentence2:str, prompt_template: str, client: LLMClient, max_retries: int) -> Optional[Dict]:
    prompt = prompt_template.format(sentence_1=sentence1, sentence_2=sentence2)
    messages = [
        {"role": "system", "content": "You are a sentence similarity rating classifier."},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat_completion(messages, temperature=0.3)
            rating = parse_output(response)
            if rating['label'] == '0' or rating['label'] == '1':
                return int(rating['label'])
            else:
                raise Exception("Label neither 0 or 1")
        except Exception as e:
                print(f"Retry {attempt + 1} for pair due to error: {e}")
    return None


# Post-processing Functions
def save_output(output_data: dict, file_path: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_cleaned_w_reason.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved Cleaned QA Pairs to: {output_path}")
    return output_path



# ============================ MAIN Function =============================== #

def qa_cleaning_batch(pairs: List[dict], prompt_template: str, client: LLMClient, verbose: bool) -> List[Optional[int]]:
    """Process a batch of QA pairs for cleaning."""
    messages = []
    for pair in pairs:
        prompt = prompt_template.format(chunk=pair['chunk'], question=pair['question'], answer=pair['answer'])
        messages.append([
            {"role": "system", "content": "You are a question answers rating system which is provided with an additional passage to enhance and help with the rating."},
            {"role": "user", "content": prompt}
        ])

    try:
        responses = client.batch_completion(messages, temperature=0.3, batch_size=len(messages))
        ratings = []
        for i, response in enumerate(responses):
            try:
                rating = parse_output(response)
                ratings.append(int(rating['rating']))
            except Exception as e:
                if verbose:
                    print(f"Error parsing rating for pair {i}: {e}")
                ratings.append(None)
        return ratings
    except Exception as e:
        if verbose:
            print(f"Batch processing error: {e}")
        return [None] * len(pairs)

def question_response_batch(questions: List[str], client: LLMClient, verbose: bool) -> List[Optional[str]]:
    """Process a batch of questions for synthetic answers."""
    messages = [
        [
            {"role": "system", "content": """Answer in a short manner and give your final answer within a json as follows {"answer":<your answer>}"""},
            {"role": "user", "content": question}
        ]
        for question in questions
    ]

    try:
        responses = client.batch_completion(messages, temperature=0.3, batch_size=len(messages))
        answers = []
        for i, response in enumerate(responses):
            try:
                output = parse_output(response)
                answers.append(output['answer'])
            except Exception as e:
                if verbose:
                    print(f"Error parsing answer for question {i}: {e}")
                answers.append(None)
        return answers
    except Exception as e:
        if verbose:
            print(f"Batch processing error: {e}")
        return [None] * len(questions)

def qa_filtering_batch(sentence_pairs: List[tuple], prompt_template: str, client: LLMClient, verbose: bool) -> List[Optional[int]]:
    """Process a batch of sentence pairs for filtering."""
    messages = [
        [
            {"role": "system", "content": "You are a sentence similarity rating classifier."},
            {"role": "user", "content": prompt_template.format(sentence_1=s1, sentence_2=s2)}
        ]
        for s1, s2 in sentence_pairs
    ]

    try:
        responses = client.batch_completion(messages, temperature=0.3, batch_size=len(messages))
        labels = []
        for i, response in enumerate(responses):
            try:
                rating = parse_output(response)
                if rating['label'] in ['0', '1']:
                    labels.append(int(rating['label']))
                else:
                    if verbose:
                        print(f"Invalid label for pair {i}")
                    labels.append(None)
            except Exception as e:
                if verbose:
                    print(f"Error parsing label for pair {i}: {e}")
                labels.append(None)
        return labels
    except Exception as e:
        if verbose:
            print(f"Batch processing error: {e}")
        return [None] * len(sentence_pairs)

def process_file(
    file_path: str,
    config_path: Optional[Path] = None,
    api_base: Optional[str] = None,
    threshold: Optional[int] = None,
    model: Optional[str] = None,
    output_dir: str = "data/cleaned",
    verbose: bool = False,
    max_retries: int = 3,
    action: str = "clean_and_filter"  # Options: "clean_only", "clean_and_filter"
) -> str:
    """
    Processes a QA file by cleaning and optionally filtering the QA pairs in batches.

    Args:
        file_path (str): Path to the input file.
        config_path (Optional[Path]): Path to the configuration file.
        api_base (Optional[str]): Optional API base URL.
        threshold (Optional[int]): Filtering threshold. Defaults to config value if not provided.
        model (Optional[str]): Model name for API calls.
        output_dir (str): Directory where cleaned data is saved.
        verbose (bool): If True, prints debug information.
        max_retries (int): Max retry attempts for API calls.
        action (str): "clean_only" to only clean, "clean_and_filter" to clean and filter.

    Returns:
        str: Path to the saved output file.
    """
    if action not in {"clean_only", "clean_and_filter"}:
        raise ValueError(f"Invalid action: {action}. Use 'clean_only' or 'clean_and_filter'.")

    # Load QA pairs and initialize
    qa_pairs, summary, all_chunks = load_qa_data(file_path)
    if verbose:
        print(f"Loaded {len(qa_pairs)} QA pairs")

    client, clean_prompt_template, filter_prompt_template, curate_config = initialize_client(config_path, api_base, model)
    if threshold is None:
        threshold = int(curate_config.get("threshold", 7.0))

    all_clean_qa = []

    # Step 1: Clean all QA pairs in a single batch
    if verbose:
        print("Processing cleaning batch...")
    
    ratings = qa_cleaning_batch(qa_pairs, clean_prompt_template, client, verbose)
    
    if action == "clean_and_filter":
        # Step 2: Generate synthetic answers for all questions in a single batch
        if verbose:
            print("Generating synthetic answers batch...")
        
        # Only process questions for pairs that passed the cleaning threshold
        valid_pairs = [(i, pair) for i, (pair, rating) in enumerate(zip(qa_pairs, ratings)) 
                        if rating is not None and rating >= threshold]
        
        if valid_pairs:
            valid_indices, valid_qa_pairs = zip(*valid_pairs)
            questions = [pair['question'] for pair in valid_qa_pairs]
            synthetic_answers = question_response_batch(questions, client, verbose)

            # Step 3: Filter using sentence comparison in a single batch
            if verbose:
                print("Processing filtering batch...")
            
            sentence_pairs = [(pair['answer'], syn_ans) 
                            for pair, syn_ans in zip(valid_qa_pairs, synthetic_answers)
                            if syn_ans is not None]
            
            if sentence_pairs:
                filter_labels = qa_filtering_batch(sentence_pairs, filter_prompt_template, client, verbose)
                
                # Collect final results
                for (idx, pair), label in zip(valid_pairs, filter_labels):
                    if label == 0:  # Only keep pairs that are sufficiently different
                        all_clean_qa.append(pair)
                        if verbose:
                            print(f"Accepted pair {idx + 1} (Rating: {ratings[idx]}, Label: {label})")
                    elif verbose:
                        print(f"Rejected pair {idx + 1} (Rating: {ratings[idx]}, Label: {label})")
    else:
        # For clean_only, just use the ratings
        for i, (pair, rating) in enumerate(zip(qa_pairs, ratings)):
            if rating is not None and rating >= threshold:
                all_clean_qa.append(pair)
                if verbose:
                    print(f"Accepted pair {i + 1} (Rating: {rating})")
            elif verbose:
                print(f"Rejected pair {i + 1} (Rating: {rating})")

    if verbose:
        print(f"\nProcessing complete:")
        print(f"- Total pairs processed: {len(qa_pairs)}")
        print(f"- Valid pairs saved: {len(all_clean_qa)}")
        print(f"- Success rate: {(len(all_clean_qa)/len(qa_pairs))*100:.1f}%")

    # Save results
    output_data = {
        "summary": summary,
        "qa_pairs": all_clean_qa,
        "all_chunks": all_chunks,
    }
    
    return save_output(output_data, file_path, output_dir)

