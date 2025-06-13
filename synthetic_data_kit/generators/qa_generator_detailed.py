from typing import Dict, List, Any, Optional, Union
import json
import os
import random
import re
from pathlib import Path

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.config import load_config, get_generation_config, get_prompt

class QAGenerator:
    def __init__(self, client: LLMClient, config_path: Optional[Path] = None):
        self.client = client
        self.config = load_config(config_path)
        self.generation_config = get_generation_config(self.config)
        self.verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'

    def debug(self, message: str):
        if self.verbose:
            print(f"[DEBUG] {message}")

    #------------------------- Sentence Splitting and Chunking -------------------------

    def split_and_mark_sentences(self, text: str, chunk_size: int = 4000) -> List[str]:
        all_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        total_sentences = len(all_sentences)
        self.debug(f"Total sentences in text: {total_sentences}")

        paragraphs = text.split('\n\n')
        chunks, current_chunk, current_sentence_count = [], "", 0

        for para in paragraphs:
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 > chunk_size:
                        chunks.append(self._mark_sentences_in_chunk(current_chunk, current_sentence_count))
                        current_sentence_count += len(re.split(r'(?<=[.!?])\s+', current_chunk.strip()))
                        current_chunk = sentence
                    else:
                        current_chunk += " " + sentence
            else:
                if len(current_chunk) + len(para) + 2 > chunk_size:
                    chunks.append(self._mark_sentences_in_chunk(current_chunk, current_sentence_count))
                    current_sentence_count += len(re.split(r'(?<=[.!?])\s+', current_chunk.strip()))
                    current_chunk = para
                else:
                    current_chunk += ("\n\n" if current_chunk else "") + para

        if current_chunk.strip():
            chunks.append(self._mark_sentences_in_chunk(current_chunk, current_sentence_count))

        self.debug(f"Generated {len(chunks)} sentence-marked chunks")
        return chunks

    def _mark_sentences_in_chunk(self, chunk: str, start_count: int) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', chunk.strip())
        return ' '.join([f"[S{start_count + i + 1}] {s}" for i, s in enumerate(sentences) if s])

    def merge_chunk_pairs(self, chunks: List[str]) -> List[List[str]]:
        if len(chunks) <= 1:
            return [chunks]
        random.shuffle(chunks)
        merged = [chunks[i:i + 2] for i in range(0, len(chunks), 2)]
        self.debug(f"Merged chunks into {len(merged)} pairs")
        return merged

    #------------------------- JSON Parsing and LLM Output Handling -------------------------

    def parse_json_from_response(self, response: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        try:
            match = re.search(r'```json\s*(\[\s*{.*?}\s*\])\s*```', response, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise ValueError("No valid JSON block found.")
        except Exception as e:
            raise ValueError(f"Invalid JSON format: {e}")

    def parse_llm_outputs(self, outputs: List[str], chunk_inputs: List[List[str]]) -> List[Dict[str, Any]]:
        all_qa_pairs = []
        for i, (output, chunk) in enumerate(zip(outputs, chunk_inputs)):
            try:
                qa_list = self.parse_json_from_response(output)
                if not isinstance(qa_list, list):
                    raise ValueError("Parsed data is not a list of QA pairs.")

                passages = "\n\n".join([f"Passage {idx + 1}:\n{p}" for idx, p in enumerate(chunk)])

                for qa in qa_list:
                    if not isinstance(qa, dict) or "question" not in qa or "answer" not in qa:
                        raise ValueError("Invalid QA format.")
                    qa.update({"chunk": passages, "chunk_list": chunk})
                    # self.debug(f"QA PAIR GENERATED:\n{qa['question']}\n{qa['answer']}")
                    all_qa_pairs.append(qa)
            except Exception as e:
                self.debug(f"Skipping a sample due to parsing error: {e} \nOutput: {output}")
        return all_qa_pairs

    #------------------------- Main Generation Methods -------------------------

    def generate_qa_pairs(self, document_text: str, summary: str, num_pairs: int = 25) -> List[Dict[str, str]]:
        chunk_size = self.generation_config.get("chunk_size", 4000)
        temperature = self.generation_config.get("temperature", 0.7)

        chunks = self.split_and_mark_sentences(document_text, chunk_size=chunk_size)
        chunk_inputs = self.merge_chunk_pairs(chunks)
        self.debug(f"Document split into {len(chunks)} chunks, merged into {len(chunk_inputs)} inputs")

        prompt_template = get_prompt(self.config, "qa_generation_detailed")
        messages = []
        for chunk in chunk_inputs:
            passages = "\n\n".join([f"Passage {i+1}:\n{p}" for i, p in enumerate(chunk)])
            system_content = prompt_template.format(num_pairs=num_pairs, summary=summary[:100], text=passages)
            messages.append([{"role": "system", "content": system_content}])

        responses = self.client.batch_completion(messages, temperature=temperature, batch_size=len(messages))
        return self.parse_llm_outputs(responses, chunk_inputs)

    def generate_summary(self, document_text: str) -> str:
        self.debug("Generating document summary...")
        prompt = get_prompt(self.config, "summary")
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": document_text + "\no_think"}
        ]
        summary = self.client.chat_completion(messages, temperature=0.1)
        self.debug(f"Summary generated ({len(summary)} chars)")
        return summary

    #------------------------- Main Processing Method -------------------------

    def process_document(self, input_file_path: str, num_pairs: int = 25, output_dir: Optional[str] = None, verbose: bool = False) -> str:
        self.verbose = verbose

        # Extract file name (without extension)
        file_name = os.path.splitext(os.path.basename(input_file_path))[0]

        # Read the document text from file
        with open(input_file_path, "r", encoding="utf-8") as file:
            document_text = file.read()

        # Process the document
        summary = self.generate_summary(document_text)
        qa_pairs = self.generate_qa_pairs(document_text, summary, num_pairs)
        marked_text = self.split_and_mark_sentences(document_text)

        result = {
            "qa_pairs": qa_pairs,
            "all_chunks": marked_text
        }

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save output with the same name as input file
        output_path = os.path.join(output_dir, f"{file_name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        self.debug(f"Saved QA output to {output_path}")
        return output_path
