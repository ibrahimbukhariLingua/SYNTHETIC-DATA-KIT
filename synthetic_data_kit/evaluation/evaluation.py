# Basic imports
import os, re, json, random
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm

# Torch and HuggingFace imports
import torch
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Set the environment variable for HuggingFace token
from huggingface_hub.hf_api import HfFolder
HfFolder.save_token("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")

# ---------------------- Utility Functions ----------------------

def set_seed(seed: int = 42):
    """
    Set seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extract and parse the first JSON object found in the LLM response.

    Supports JSON enclosed in triple backticks or raw JSON.
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
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")


# ---------------------- Preprocessor ----------------------

class DatasetPreprocessor:
    """
    Provides methods for formatting examples and cleaning model responses.
    """

    @staticmethod
    def context_format(dataset_title:str, context: str) -> str:
        """
        Format the context by removing excessive whitespace and ensuring clean text.
        """
        
        # adds sentence markers: [P_][S_]
        def process_passages(source_text):
            def tag_sentences(passage_tag, passage_text):
                # Extract passage number or use 'Pn' for generic names
                match = re.match(r'passage(\d+)', passage_tag)
                passage_number = f'P{match.group(1)}' if match else 'Pn'
                
                # Split into sentences
                sentences = re.split(r'(?<=\.)\s+', passage_text.strip())
                tagged_sentences = []
                for i, sentence in enumerate(sentences, 1):
                    sent_number = f'S{i}'
                    tagged_sentences.append(f'[{passage_number}][{sent_number}]{sentence}')
                
                return f'<{passage_tag}> ' + ' '.join(tagged_sentences) + f' </{passage_tag}>'

            # Find all passages
            pattern = re.compile(r'<(passage\w+)>(.*?)</\1>', re.DOTALL)
            processed = []

            for match in pattern.finditer(source_text):
                tag, content = match.group(1), match.group(2)
                processed.append(tag_sentences(tag, content))

            return '\n'.join(processed)
        
        # formats the raw text into within a passage tag
        def passage_format(passage: str) -> str:
            pattern = r"Passage (\d+):\s+(.*?)(?=Passage \d+:|$)"
            matches = re.findall(pattern, passage, re.DOTALL)
            
            formatted_passages = []
            for number, text in matches:
                tag = f"passage{number}"
                formatted_passages.append(f"<{tag}> {text.strip()} </{tag}>")
            
            return "\n".join(formatted_passages)
        
        def chapter_format(raw_text: str) -> str:
            # Pattern to find chapter headers like I, II, III, IV, etc. (Roman numerals)
            chapter_pattern = r'(?<=\n)([IVXLCDM]+)\n'
            
            # Find all chapter starts
            matches = list(re.finditer(chapter_pattern, raw_text))
            
            if not matches:
                # No chapters found, treat whole text as passage1
                return f"<passage1> {raw_text.strip()} </passage1>"
            
            passages = []
            last_end = 0

            # Handle everything before the first chapter as passage1
            first_match = matches[0]
            pre_chapter_text = raw_text[:first_match.start()].strip()
            if pre_chapter_text:
                passages.append(f"<passage1> {pre_chapter_text} </passage1>")
                passage_num = 2
            else:
                passage_num = 1

            # Process each chapter section
            for i, match in enumerate(matches):
                chapter_title = match.group(1)
                start = match.end()

                # Determine end of this chapter
                end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)

                chapter_text = raw_text[start:end].strip()
                passages.append(f"<passage{passage_num}> {chapter_text} </passage{passage_num}>")
                passage_num += 1

            return '\n\n'.join(passages)
        
        def simple_format(text: str) -> str:
            # Split based on paragraph breaks: two or more newlines OR newline followed by capital letter
            paragraphs = re.split(r'\n\s*\n+|\n(?=[A-Z])', text.strip())
            
            formatted_passages = []
            for i, para in enumerate(paragraphs, 1):
                para = para.strip()
                if para:  # Ignore empty strings
                    formatted_passages.append(f"<passage{i}>{para}</passage{i}>")
            
            return "\n".join(formatted_passages)
        
        # Choose the appropriate formatting function
        if dataset_title in ["2wikimqa_e", "musique"]:
            formatted = process_passages(passage_format(context))
        elif dataset_title == "narrativeqa":
            formatted = process_passages(chapter_format(context))
        elif dataset_title == "multifieldqa_en":
            formatted = process_passages(simple_format(context))
        else:
            raise ValueError(f"Unsupported dataset title: {dataset_title}")

        # Sanity check for passage tags
        if not re.search(r"<passage\d+>", formatted):
            raise ValueError("Formatting failed: No <passage> tags found in output.")

        return formatted

    @staticmethod
    def format_input(example: Dict, dataset_title:str) -> Dict[str, str]:
        """
        Convert a raw dataset example into a prompt + reference format.
        """
        question = example["input"].strip()
        context = example["context"].strip()
        tagged_context = DatasetPreprocessor.context_format(dataset_title, context)
        answers = example.get("answers", [])
        prompt = f"Source Passage:\n{tagged_context}\n\nUser Question:\n{question}"
        reference = answers[0].strip() if isinstance(answers, list) and answers else ""
        return {
            "prompt": prompt,
            "reference": reference,
            "context": context
        }

    @staticmethod
    def clean_response(response: str) -> str:
        """
        Strip <think> tags and their contents from a model's response.
        """
        return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


# ---------------------- Model Evaluator ----------------------

class ModelEvaluator:
    def __init__(self, model_name: str, max_new_tokens: int = 2000, batch_size: int = 4, verbose: bool = False):
        set_seed(42)
        self.model_name = model_name
        self.verbose = verbose
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

        # ====== Load Tokenizer & Model with Flash Attention ======
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
            # attn_implementation="flash_attention_2",  # ⚡ Enable Flash Attention 2
            trust_remote_code=True
        )
        self.model.eval()

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )

    # ---------------------- Helper Functions ---------------------- #
    
    def log(self, message: str):
        if self.verbose:
            print(f"[DEBUG] {message}")

    def apply_template(self, prompt: str) -> List[Dict[str, str]]:
        # system_prompt = (
        #     "Format Example:"
        #     "<statement>This is sentence 1.<cite>[P1][S4]</cite></statement>"
        #     "<statement>This is sentence 2.<cite></cite></statement>"
        #     "You are a question answering system that uses a provided Source passage to answer user questions."
        #     "Each sentence in your response must be wrapped in <statement> tags."
        #     "At the end of each sentence, include a <cite> tag that lists the source sentence numbers used to support that statement in the format [P_|S_][P_|S_]."
        #     "If no source is used for a sentence, leave the <cite> tag empty."
        # )
        system_prompt = (
            "You are a question answering system which will be provided with a Source passage. "
            "Using the Source passage you will answer the user's question. "
            "You will cite sources which you use when answering the question."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

    def evaluate_response_manually(self, reference: str, generated: str) -> int:
        clean_generated = re.sub(r'</?(statement|cite)>', '', generated).lower()
        reference = reference.lower()
        return 1 if reference in clean_generated else 0

    # ---------------------- Response Generation ---------------------- #

    #========================
    # @torch.inference_mode()
    def batch_generate_responses(self, batched_messages: List[List[Dict[str, str]]]) -> List[str]:
        prompts = [
            self.tokenizer.apply_chat_template(dialogue, tokenize=False, add_generation_prompt=True)
            for dialogue in batched_messages
        ]

        outputs = self.generator(
            prompts,
            max_new_tokens=self.max_new_tokens,
            return_full_text=False,
            batch_size=self.batch_size,
            truncation=True
        )

        return [
            DatasetPreprocessor.clean_response(out[0]["generated_text"])
            for out in outputs
        ], [out[0]["generated_text"] for out in outputs]

    def process_batch(self, examples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        prompts = [ex["prompt"] for ex in examples]
        references = [ex["reference"] for ex in examples]
        contexts = [ex["context"] for ex in examples]

        batched_inputs = [self.apply_template(p) for p in prompts]
        generated_responses, raw_responses = self.batch_generate_responses(batched_inputs)

        results = []
        for prompt, reference, context, generated, raw in zip(prompts, references, contexts, generated_responses, raw_responses):
            label = self.evaluate_response_manually(reference, generated)
            self.log(f"Prompt Length: {len(prompt.split())}")
            self.log(f"Generated Answer: {generated}")
            self.log(f"Reference Answer: {reference}")
            self.log(f"Label: {label}")

            results.append({
                "dataset": self.model_name,
                "question": prompt,
                "reference_answer": reference,
                "context": context,
                "generated_answer": raw,
                "label": label
            })
        return results

    # ---------------------- Dataset Evaluation ---------------------- #

    def evaluate_dataset(self, dataset_name: str, split: str = "test") -> Dict:
        dataset = load_dataset("THUDM/LongBench", dataset_name, split=split)
        processed_examples = [
            DatasetPreprocessor.format_input(example=ex, dataset_title=dataset_name)
            for ex in dataset
        ]

        evaluated_results = []
        correct_predictions = 0

        for i in tqdm(range(0, len(processed_examples), self.batch_size), desc=f"Evaluating {dataset_name}"):
            batch = processed_examples[i:i+self.batch_size]
            try:
                batch_results = self.process_batch(batch)
                evaluated_results.extend(batch_results)
                correct_predictions += sum(r["label"] for r in batch_results)
            except Exception as e:
                self.log(f"Batch error at index {i}: {e}")
            
            torch.cuda.empty_cache() 
            self.log("Batch processing complete.\n============================================================================")

        accuracy = correct_predictions / len(evaluated_results) if evaluated_results else 0
        metrics = {"accuracy": accuracy}
        final_results = {
            "all_responses": evaluated_results,
            "metrics": metrics
        }

        safe_model_name = self.model_name.replace("/", "_")
        output_dir = f"/data/home/syed.bukhari/synthetic-data-kit/data/results/{safe_model_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{dataset_name}.jsonl")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False)

        print(f"Saved results to {output_path}")
        return {
            "dataset": dataset_name,
            "samples": len(evaluated_results),
            "output_file": output_path,
            "metrics": metrics
        }





