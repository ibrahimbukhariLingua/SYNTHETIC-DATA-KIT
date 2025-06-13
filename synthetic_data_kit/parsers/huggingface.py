# import os
# import json
# import fitz  # PyMuPDF

# class FinanceBenchGitProcessor:
#     def __init__(self, repo_root):
#         self.repo_root = repo_root
#         self.questions_path = os.path.join(repo_root, "data", "financebench_open_source.jsonl")
#         self.meta_path = os.path.join(repo_root, "data", "financebench_document_information.jsonl")
#         self.pdf_dir = os.path.join(repo_root, "pdfs")
#         self.processed = []

#     # ===== Load JSONL files into memory =====
#     def load_data(self):
#         self.questions = [json.loads(line) for line in open(self.questions_path)]
#         self.docs_meta = {rec['doc_name']: rec for rec in map(json.loads, open(self.meta_path))}

#     # ===== Parse PDF by reading local file (one string per page) =====
#     def parse_pdf(self, doc_name):
#         pdf_path = os.path.join(self.pdf_dir, f"{doc_name}.pdf")
#         if not os.path.exists(pdf_path):
#             raise FileNotFoundError(f"{pdf_path} not found")
#         pages = []
#         with fitz.open(pdf_path) as doc:
#             for page in doc:
#                 pages.append(page.get_text())
#         return pages

#     # ===== Clean text utility =====
#     def clean(self, text):
#         return text.replace("\n", " ").replace("\t", " ").lower().strip()

#     # ===== Process entries pipeline =====
#     def process(self):
#         for q in self.questions:
#             doc_name = q['doc_name']
#             meta = self.docs_meta.get(doc_name)
#             if not meta:
#                 continue

#             # Parse PDF
#             try:
#                 pdf_pages = self.parse_pdf(doc_name)
#             except Exception as e:
#                 print(f"Skipping {doc_name}: {e}")
#                 continue

#             # Extract first evidence item
#             ev = q.get('evidence', [])
#             if not ev:
#                 continue

#             ev0 = ev[0]
#             ev_text = self.clean(ev0['evidence_text'])
#             ev_page = ev0['evidence_page_num']

#             if ev_page >= len(pdf_pages):
#                 continue

#             page_content = self.clean(pdf_pages[ev_page])

#             # Verify presence
#             if ev_text in page_content:
#                 entry = {
#                     'parsed_pdf': pdf_pages,
#                     'evidence_text': ev_text,
#                     'evidence_page_num': ev_page,
#                     'question': q['question'],
#                     'answer': q['answer']
#                 }
#                 self.processed.append(entry)

#     # ===== Output stats =====
#     def stats(self):
#         total = len(self.questions)
#         good = len(self.processed)
#         print(f"Processed {good}/{total} entries ({100*good/total:.1f}%)")

#     # ===== Run end-to-end =====
#     def run(self):
#         self.load_data()
#         self.process()
#         self.stats()


# if __name__ == "__main__":
#     root = "/data/home/syed.bukhari/financebench"  # adjust this!
#     processor = FinanceBenchGitProcessor(root)
#     processor.run()



import os
import json
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from io import BytesIO
from pdfminer.high_level import extract_text


class FinanceBenchGitProcessor:
    def __init__(self, repo_root):
        self.repo_root = repo_root
        self.questions_path = os.path.join(repo_root, "data", "financebench_open_source.jsonl")
        self.meta_path = os.path.join(repo_root, "data", "financebench_document_information.jsonl")
        self.pdf_dir = os.path.join(repo_root, "pdfs")

        self.questions = []
        self.docs_meta = {}
        self.processed_by_parser = {
            "pymupdf": [],
            "pdfminer": [],
            "pypdf2": []
        }

    # ===== Load the metadata and QA pairs from JSONL files =====
    def load_data(self):
        with open(self.questions_path, "r") as f:
            self.questions = [json.loads(line) for line in f]

        with open(self.meta_path, "r") as f:
            self.docs_meta = {json.loads(line)["doc_name"]: json.loads(line) for line in f}

    # ===== Parse PDF using PyMuPDF =====
    def parse_with_pymupdf(self, filepath):
        with fitz.open(filepath) as doc:
            return [page.get_text() for page in doc]

    # ===== Parse PDF using pdfminer.six =====
    def parse_with_pdfminer(self, filepath):
        full_text = extract_text(filepath)
        return full_text.split('\f')  # each page ends with \f

    # ===== Parse PDF using PyPDF2 =====
    def parse_with_pypdf2(self, filepath):
        with open(filepath, "rb") as f:
            reader = PdfReader(f)
            return [page.extract_text() or "" for page in reader.pages]

    # ===== Clean text helper =====
    def clean(self, text):
        return text.replace("\n", " ").replace("\t", " ").lower().strip()

    # ===== Attempt all parsers on a single item =====
    def try_parsers(self, qitem):
        doc_name = qitem.get("doc_name")
        ev_list = qitem.get("evidence", [])
        if not ev_list or not doc_name:
            return

        pdf_path = os.path.join(self.pdf_dir, f"{doc_name}.pdf")
        if not os.path.exists(pdf_path):
            return

        ev0 = ev_list[0]
        ev_text = self.clean(ev0["evidence_text"])
        ev_page = ev0["evidence_page_num"]

        parsers = {
            "pymupdf": self.parse_with_pymupdf,
            # "pdfminer": self.parse_with_pdfminer,
            # "pypdf2": self.parse_with_pypdf2,
        }

        for name, parser in parsers.items():
            try:
                pdf_pages = parser(pdf_path)

                if ev_page >= len(pdf_pages):
                    continue

                parsed_page = self.clean(pdf_pages[ev_page])
                if ev_text in parsed_page:
                    self.processed_by_parser[name].append({
                        "parsed_pdf": pdf_pages,
                        "evidence_text": ev_text,
                        "evidence_page_num": ev_page,
                        "question": qitem["question"],
                        "answer": qitem["answer"]
                    })
                    break  # Success on this parser, skip the rest

            except Exception as e:
                continue  # Try next parser

    # ===== Process all questions =====
    def process(self):
        for qitem in self.questions:
            self.try_parsers(qitem)

    # ===== Print final stats =====
    def stats(self):
        total = len(self.questions)
        print(f"\n==== Final Parsing Stats ====")
        for parser_name, items in self.processed_by_parser.items():
            count = len(items)
            print(f"{parser_name.upper()}: {count}/{total} ({100*count/total:.2f}%) successful")

    # ===== Run everything =====
    def run(self):
        print("Loading data...")
        self.load_data()

        print("Processing entries with multiple parsers...")
        self.process()

        self.stats()


if __name__ == "__main__":
    repo_root = "/data/home/syed.bukhari/financebench"  # <- Update this path accordingly
    processor = FinanceBenchGitProcessor(repo_root)
    processor.run()