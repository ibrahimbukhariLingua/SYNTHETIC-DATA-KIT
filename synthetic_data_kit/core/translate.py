import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.config import get_vllm_config

def translate_text(
    text: str,
    target_lang: str,
    llm_client: LLMClient,
    config: Dict[str, Any]
) -> str:
    """
    Translate text to target language using LLM.
    
    Args:
        text: Text to translate
        target_lang: Target language code (e.g., 'french', 'spanish')
        llm_client: LLM client instance
        config: Configuration dictionary
    
    Returns:
        Translated text
    """
    # Get translation prompt from config or use default
    translation_prompt = config.get("prompts", {}).get("translation", """
    You are a professional translator. Translate the following text to {target_lang}.
    Maintain the original meaning, tone, and formatting.
    Only output the translated text, nothing else.
    
    Text to translate:
    ---
    {text}
    ---
    """)
    
    # Format prompt with target language and text
    formatted_prompt = translation_prompt.format(
        target_lang=target_lang,
        text=text
    )
    
    # Get translation from LLM
    messages = [
        {"role": "system", "content": "You are a professional translator."},
        {"role": "user", "content": formatted_prompt}
    ]
    
    response = llm_client.chat_completion(
        messages=messages,
        temperature=0.3,  # Lower temperature for more consistent translations
        max_tokens=4000,
        top_p=0.95
    )
    
    return response.strip()

def translate_json_content(
    content: Union[Dict, List],
    target_lang: str,
    llm_client: LLMClient,
    config: Dict[str, Any]
) -> Union[Dict, List]:
    """
    Recursively translate JSON content.
    
    Args:
        content: JSON content to translate (dict or list)
        target_lang: Target language code
        llm_client: LLM client instance
        config: Configuration dictionary
    
    Returns:
        Translated JSON content
    """
    if isinstance(content, dict):
        return {
            key: translate_json_content(value, target_lang, llm_client, config)
            for key, value in content.items()
        }
    elif isinstance(content, list):
        return [
            translate_json_content(item, target_lang, llm_client, config)
            for item in content
        ]
    elif isinstance(content, str):
        return translate_text(content, target_lang, llm_client, config)
    else:
        return content

def process_file(
    input_path: str,
    output_dir: Path,
    target_lang: str,
    config_path: Optional[Path] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    verbose: bool = False
) -> str:
    """
    Process a JSON file for translation.
    
    Args:
        input_path: Path to input JSON file
        output_dir: Directory to save output
        target_lang: Target language code
        config_path: Path to config file
        api_base: VLLM API base URL
        model: Model to use
        verbose: Whether to show detailed output
    
    Returns:
        Path to output file
    """
    # Load config
    from synthetic_data_kit.utils.config import load_config
    config = load_config(config_path)
    
    # Get VLLM config
    vllm_config = get_vllm_config(config)
    api_base = api_base or vllm_config.get("api_base")
    model = model or vllm_config.get("model")
    
    # Initialize LLM client
    llm_client = LLMClient(
        config_path=config_path,
        api_base=api_base,
        model_name=model  # Changed from model to model_name
    )
    
    # Read input JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = json.load(f)
    
    # Translate content
    translated_content = translate_json_content(content, target_lang, llm_client, config)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_{target_lang}.json")
    
    # Save translated content
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(translated_content, f, ensure_ascii=False, indent=2)
    
    return output_path 