import os, json
from typing import List, Any

class JSONLParser:
    """Parser for JSONL documents"""
    
    def parse(self, file_path: str) -> List[str]:
        """Parse a PDF file into plain text
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        output = []
        try:
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        title = data.get("title", "")
                        text = data.get("text", "")
                        content = f"{title}\n\n\n{text}"
                        output.append(content)
                    except json.JSONDecodeError:
                        continue  # Skip invalid lines
            
        except ImportError:
            raise ImportError("Error Parsing JSONL file")
        
        return output

    def save_item(self, content: str, output_path: str) -> None:
        """Save a single content string to a file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def save(self, contents: List[str], output_folder: str) -> None:
        """Save each content item into a separate file in a specified folder."""
        # save_dir = os.path.join(output_folder, "jsonl_text")
        os.makedirs(output_folder, exist_ok=True)

        for idx, content in enumerate(contents):
            # Try to extract a title for filename
            title_line = content.split('\n', 1)[0].strip()
            safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else "_" for c in title_line)
            filename = f"{idx:05d}_{safe_title[:50]}.txt" if safe_title else f"{idx:05d}.txt"
            output_path = os.path.join(output_folder, filename)
            self.save_item(content, output_path)