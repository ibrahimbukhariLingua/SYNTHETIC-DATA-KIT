# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Text processing utilities
import re
import json
from typing import List, Dict, Any

def split_into_chunks(text: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
    """Split text into chunks with optional overlap and numbered sentences"""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    # sentence_counter = 1

    def number_sentences(chunk: str) -> str:
        # Split into sentences using regex to preserve punctuation
        sentences = re.split(r'(?<=[.!?])\s+', chunk.strip())
        numbered = [f"[S{0 + i}] {s}" for i, s in enumerate(sentences, 1)]
        return ' '.join(numbered)

    for para in paragraphs:
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            numbered_chunk = number_sentences(current_chunk)
            chunks.append(f"Chunk:\n\n{numbered_chunk}")
            # Create overlap
            sentences = re.split(r'(?<=[.!?])\s+', current_chunk.strip())
            if len(sentences) > 3:
                current_chunk = ' '.join(sentences[-3:]) + "\n\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    if current_chunk:
        numbered_chunk = number_sentences(current_chunk)
        chunks.append(f"Chunk:\n\n{numbered_chunk}")

    return chunks

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON from text that might contain markdown or other content"""
    text = text.strip()
    
    # Try to parse as complete JSON
    if text.startswith('{') and text.endswith('}') or text.startswith('[') and text.endswith(']'):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    
    # Look for JSON within Markdown code blocks
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Try a more aggressive pattern
    json_pattern = r'\{[\s\S]*\}|\[[\s\S]*\]'
    match = re.search(json_pattern, text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    raise ValueError("Could not extract valid JSON from the response")