import os, re, random, json
from typing import List, Tuple, Dict, Any
import hashlib, faiss, nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from itertools import groupby

from synthetic_data_kit.utils.config import get_prompt

# print("installing nltk-punkt")
# nltk.download('punkt')
# print("done installing nltk-punkt")

# ============================= Helper Functions ============================ #
def transform_input(text):
    def compress_citations(citations):
        matches = re.findall(r'\[P(\d+)\|S(\d+)\]', citations)
        passage_dict = {}
        for p, s in matches:
            p = int(p)
            s = int(s)
            passage_dict.setdefault(p, []).append(s)

        compressed = []
        for p in sorted(passage_dict.keys()):
            sentences = sorted(passage_dict[p])
            for k, g in groupby(enumerate(sentences), lambda ix: ix[0] - ix[1]):
                group = list(map(lambda x: x[1], g))
                if len(group) > 1:
                    compressed.append(f"[P{p}|S{group[0]}-S{group[-1]}]")
                else:
                    compressed.append(f"[P{p}|S{group[0]}]")
        return ''.join(compressed)

    def extract_statements(text):
        pattern = re.compile(r'<statement>(.*?)<cite>(.*?)</cite></statement>', re.DOTALL)
        return pattern.findall(text)

    def preprocess_statements(statements):
        processed = []
        for content, citation in statements:
            content = content.strip()
            if not citation.strip():
                processed.append((content, None))
            else:
                compressed = compress_citations(citation)
                processed.append((content, compressed))
        return processed

    def merge_statements(processed):
        merged_statements = []
        i = 0
        while i < len(processed):
            content, citation = processed[i]
            j = i + 1
            while j < len(processed) and processed[j][1] == citation:
                content += " " + processed[j][0]
                j += 1
            if citation:
                merged_statements.append(f"<statement>{content}<cite>{citation}</cite></statement>")
            else:
                merged_statements.append(content)
            i = j
        return merged_statements

    def remove_statement_tags(text):
        return re.sub(r'</?statement>', '', text)

    # === Pipeline ===

    # Step 1: Extract statements
    statements = extract_statements(text)

    # Step 2: Preprocess statements (can comment out to skip compression)
    statements = preprocess_statements(statements)

    # Step 3: Merge consecutive statements with same citations (optional)
    output = merge_statements(statements)

    # Step 4: Remove statement tags (optional)
    output = [remove_statement_tags(stmt) for stmt in output]

    return ''.join(output)

def generate_ft_model_name(model: str, input_dir: str, num_of_samples: int, version: str = "v3") -> str:
    """
    Generate a fine-tuned model name based on the model path, input file, sample size, and version.

    Args:
        model (str): The base model path or name (e.g., "Qwen/Qwen1.5-7B-Chat").
        input_dir (str): Path to the input file (e.g., "data/my_dataset.txt").
        num_of_samples (int): Number of samples used for fine-tuning.
        version (str): Optional version tag. Defaults to "v3".

    Returns:
        str: The constructed fine-tuned model name.
    """
    filename = os.path.basename(input_dir)
    if ".txt" in filename:
        dataset_name = filename.split(".txt", 1)[0] + filename.split(".txt", 1)[1]
    else:
        dataset_name = os.path.splitext(filename)[0]

    model_base = model.split("/")[-1].lower().replace("-instruct", "").replace("-chat", "").replace("_", "-")
    ft_model_name = f"{model_base}-{dataset_name}-{num_of_samples}-{version}"
    return ft_model_name




# ============================ Finetuning Data Formatting ========================== #
def split_into_chunks(text: str, chunk_size: int = 4000) -> List[str]:
    """
    Splits a long text into chunks of approximately `chunk_size` characters,
    preferably on sentence boundaries if possible.
    """
    import re

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


class RAGFormatter:
    def __init__(self, directory: str):
        # ---------------------------------------- Initialization ------------------------------------
        print("Initializing RAGFormatter...")
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)
        os.makedirs("data/index", exist_ok=True)

        print(f"Loading model from 'all-MiniLM-L6-v2'...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.texts = self._load_texts() 
        self.index_path = self._get_index_path()

        if os.path.exists(self.index_path):
            print(f"Loading existing index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
        else:
            print(f"Creating new index at {self.index_path}")
            self.embeddings = self._embed_texts()
            self.index = self._build_index()
            faiss.write_index(self.index, self.index_path)

    def _get_index_path(self) -> str:
        dir_hash = hashlib.md5(self.directory.encode()).hexdigest()
        return os.path.join("data/index", f"index_{dir_hash}.faiss")

    # ---------------------------------------- File Loading & Embedding ------------------------------------

    def _load_texts(self) -> List[str]:
        files = sorted([f for f in os.listdir(self.directory) if f.endswith('.txt')])
        return [open(os.path.join(self.directory, f), encoding='utf-8').read() for f in files]

    def _embed_texts(self) -> np.ndarray:
        return np.array(self.model.encode(self.texts, show_progress_bar=True, convert_to_numpy=True))

    def _build_index(self):
        index = faiss.IndexFlatL2(self.embeddings.shape[1])
        index.add(self.embeddings)
        return index

    # ---------------------------------------- Retrieval ------------------------------------

    def _retrieve_top_k(self, question: str, gold_passages: List[str], k: int) -> List[str]:
        cleaned_gold_passages = [
            re.sub(r'\[S\d+\]\s*', '', gp).strip()
            for gp in gold_passages
        ]

        q_vec = self.model.encode([question], convert_to_numpy=True)
        _, indices = self.index.search(q_vec, len(self.texts))

        retrieved = []
        for i in indices[0]:
            candidate = self.texts[i]
            if all(cleaned not in candidate for cleaned in cleaned_gold_passages):
                retrieved.append(candidate)
            if len(retrieved) == k:
                break
        return retrieved

    # ---------------------------------------- Formatting ------------------------------------

    def _renumber_and_format_gold_passage(self, passage: str, passage_number: int) -> Tuple[str, Dict[str, str]]:
        old_tags = re.findall(r'\[S\d+\]', passage)
        new_passage = passage
        tag_map = {}
        for i, old_tag in enumerate(old_tags):
            new_tag = f"|S{i+1}]"
            tag_map[old_tag] = f"[P{passage_number}{new_tag}"
            new_passage = new_passage.replace(old_tag, f"[P{passage_number}{new_tag}", 1)
        return new_passage, tag_map

    def _format_regular_passage(self, passage: str, passage_number: int) -> str:
        sentences = nltk.sent_tokenize(passage)
        return " ".join([f"[P{passage_number}|S{s+1}] {sent}" for s, sent in enumerate(sentences)])

    # ---------------------------------------- Citation Handling ------------------------------------

    def _update_gold_text(self, text: str, marker_map: Dict[str, str]) -> str:
        def repl(match):
            tags = re.findall(r'\[S\d+\]', match.group(0))
            return ''.join([marker_map.get(tag, tag) for tag in tags])
        return re.sub(r'((\[S\d+\]){1,})', repl, text)

    # ---------------------------------------- Public API ------------------------------------

    def format_for_question(
        self,
        question: str,
        gold_passages: List[str],
        gold_answer: str,
        gold_thinking: str,
        top_k: int = 3
    ) -> Tuple[List[str], str, str]:
        retrieved_texts = self._retrieve_top_k(question, gold_passages, k=top_k)

        insert_positions = sorted(random.sample(range(len(retrieved_texts) + len(gold_passages)), len(gold_passages)))

        passages_with_gold = []
        gold_passage_numbers = []
        gold_index = 0
        text_index = 0

        full_marker_map = {}

        for i in range(len(retrieved_texts) + len(gold_passages)):
            if i in insert_positions:
                passage_number = i + 1
                gold_passage = gold_passages[gold_index]
                formatted_gold, marker_map = self._renumber_and_format_gold_passage(gold_passage, passage_number)
                passages_with_gold.append(formatted_gold)
                gold_passage_numbers.append(passage_number)
                full_marker_map.update(marker_map)
                gold_index += 1
            else:
                passage_number = i + 1
                formatted = self._format_regular_passage(retrieved_texts[text_index], passage_number)
                passages_with_gold.append(formatted)
                text_index += 1

        updated_gold_answer = self._update_gold_text(gold_answer, full_marker_map)
        updated_gold_thinking = self._update_gold_text(gold_thinking, full_marker_map)

        return passages_with_gold, updated_gold_answer, updated_gold_thinking


