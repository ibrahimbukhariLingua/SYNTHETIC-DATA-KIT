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




"""
Evaluating Tags format:

- takes which tags to evaluate check for
- has a check tag function that checks if the tags are present in a single answer
- theis is used in evaluate_tags function which loops over each answer
"""

class EvaluateTags:
    def __init__(self):
        """
        To DO:
        
        - has a dictionary where keys are the tag names and the value is a list which consists of the oprning tag and closing tag
        """
        pass
    
    def check_tag(self, answer: str) -> bool:
        """
        To Do:
        
        - checks if the tags from our constructor are present in the answer
        - if statement tag is in the tag dictionary keys then check if it is properly opening and closing around a single sentence
        - if cite tag is in the tag dictionary keys, it should properly open and close. it should be within the statement tag. it should be at the end of the sentence. it should not be empty
        - for now the logic to each tag should be within a small function inside this function. so we can easily modify the logic later if needed.
        - if all the tags are present and properly formatted, return 1, else return 0
        """
        pass




# Evaluating Citations 
