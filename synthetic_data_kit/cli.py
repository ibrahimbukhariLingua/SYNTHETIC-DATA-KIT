# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# CLI Logic for synthetic-data-kit

import os, re
import typer
from pathlib import Path
from typing import Optional
import requests
from rich.console import Console
from rich.table import Table
import json

from synthetic_data_kit.utils.config import load_config, get_vllm_config, get_path_config
from synthetic_data_kit.core.context import AppContext

# Initialize Typer app
app = typer.Typer(
    name="synthetic-data-kit",
    help="A toolkit for preparing synthetic datasets for fine-tuning LLMs",
    add_completion=True,
)
console = Console()

# Create app context
ctx = AppContext()

# Check Server
def check_server(api_base, api_key, model, console):
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.get(f"{api_base}/models", headers=headers, timeout=2)
        if response.status_code != 200:
            console.print(f"L Error: VLLM server not available at {api_base} (Status {response.status_code})", style="red")
            console.print("Please start the VLLM server with:", style="yellow")
            console.print(f"vllm serve {model}", style="bold blue")
            return 1
    except requests.exceptions.RequestException as e:
        console.print(f"L Error: VLLM server not available at {api_base}", style="red")
        console.print(f"Exception: {e}", style="red")
        console.print("Please start the VLLM server with:", style="yellow")
        console.print(f"vllm serve {model}", style="bold blue")
        return 1

    return 0

# Define global options
@app.callback()
def callback(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
):
    """
    Global options for the Synthetic Data Kit CLI
    """
    if config:
        ctx.config_path = config
    ctx.config = load_config(ctx.config_path)


@app.command("system-check")
def system_check(
    api_base: Optional[str] = typer.Option(
        None, "--api-base", help="VLLM API base URL to check"
    )
):
    """
    Check if the VLLM server is running.
    """
    # Get VLLM server details from args or config
    vllm_config = get_vllm_config(ctx.config)
    api_base = api_base or vllm_config.get("api_base")
    
    with console.status(f"Checking VLLM server at {api_base}..."):
        try:
            response = requests.get(f"{api_base}/models", timeout=2)
            if response.status_code == 200:
                console.print(f" VLLM server is running at {api_base}", style="green")
                console.print(f"Available models: {response.json()}")
                return 0
            else:
                console.print(f"L VLLM server is not available at {api_base}", style="red")
                console.print(f"Error: Server returned status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            console.print(f"L VLLM server is not available at {api_base}", style="red")
            console.print(f"Error: {str(e)}")
            
        # Show instruction to start the server
        model = vllm_config.get("model")
        port = vllm_config.get("port", 8000)
        console.print("\nTo start the server, run:", style="yellow")
        console.print(f"vllm serve {model} --port {port}", style="bold blue")
        return 1


@app.command()
def ingest(
    input: str = typer.Argument(..., help="File or URL to parse"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Where to save the output"
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Custom output filename"
    ),
):
    """
    Parse documents (PDF, HTML, YouTube, DOCX, PPT, TXT) into clean text.
    """
    from synthetic_data_kit.core.ingest import process_file
    
    # Get output directory from args, then config, then default
    if output_dir is None:
        output_dir = get_path_config(ctx.config, "output", "parsed")
    
    try:
        with console.status(f"Processing {input}..."):
            output_path = process_file(input, output_dir, name, ctx.config)
        console.print(f" Text successfully extracted to [bold]{output_path}[/bold]", style="green")
        return 0
    except Exception as e:
        console.print(f"L Error: {e}", style="red")
        return 1


@app.command()
def create(
    input: str = typer.Argument(..., help="File to process"),
    content_type: str = typer.Option(
        "qa", "--type", help="Type of content to generate [qa|summary|cot|cot-enhance]"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Where to save the output"
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base", help="VLLM API base URL"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="VLLM API key"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use"
    ),
    num_pairs: Optional[int] = typer.Option(
        None, "--num-pairs", "-n", help="Target number of QA pairs to generate"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
):
    """
    Generate content from text using local LLM inference.
    
    Content types:
    - qa: Generate question-answer pairs from text
    - summary: Generate a summary of the text
    - cot: Generate Chain of Thought reasoning examples from text
    - cot-enhance: Enhance existing tool-use conversations with Chain of Thought reasoning
      (for cot-enhance, the input must be a JSON file with either:
       - A single conversation in 'conversations' field
       - An array of conversation objects, each with a 'conversations' field
       - A direct array of conversation messages)
    """
    from synthetic_data_kit.core.create import process_file
    
    # Get VLLM server details from args or config
    vllm_config = get_vllm_config(ctx.config)
    api_base = api_base or vllm_config.get("api_base")
    api_key = api_key or vllm_config.get("api_key")
    model = model or vllm_config.get("model")
    
    # Check server first
    if check_server(api_base, api_key, model, console) != 0:
        return
    
    # Get output directory from args, then config, then default
    if output_dir is None:
        output_dir = get_path_config(ctx.config, "output", "generated")
    
    try:
        with console.status(f"Generating {content_type} content from {input}..."):
            output_path = process_file(
                input,
                output_dir,
                ctx.config_path,
                api_base,
                model,
                content_type,
                num_pairs,
                verbose
            )
        if output_path:
            console.print(f" Content saved to [bold]{output_path}[/bold]", style="green")
        return 0
    except Exception as e:
        console.print(f"L Error: {e}", style="red")
        return 1


@app.command("curate")
def curate(
    input: str = typer.Argument(..., help="Input file to clean"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    threshold: Optional[float] = typer.Option(
        None, "--threshold", "-t", help="Quality threshold (1-10)"
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base", help="VLLM API base URL"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="VLLM API key"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
):
    """
    Clean and filter content based on quality.
    """
    from synthetic_data_kit.core.curate import curate_qa_pairs
    
    # Get VLLM server details from args or config
    vllm_config = get_vllm_config(ctx.config)
    api_base = api_base or vllm_config.get("api_base")
    api_key = api_key or vllm_config.get("api_key")
    model = model or vllm_config.get("model")
    
    # Check server first
    if check_server(api_base, api_key, model, console) != 0:
        return
    
    # Get default output path from config if not provided
    if not output:
        cleaned_dir = get_path_config(ctx.config, "output", "cleaned")
        os.makedirs(cleaned_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input))[0]
        output = os.path.join(cleaned_dir, f"{base_name}_cleaned.json")
    
    try:
        with console.status(f"Cleaning content from {input}..."):
            result_path = curate_qa_pairs(
                input,
                output,
                threshold,
                api_base,
                model,
                ctx.config_path,
                verbose
            )
        console.print(f" Cleaned content saved to [bold]{result_path}[/bold]", style="green")
        return 0
    except Exception as e:
        console.print(f"L Error: {e}", style="red")
        return 1


@app.command("save-as")
def save_as(
    input: str = typer.Argument(..., help="Input file to convert"),
    format: Optional[str] = typer.Option(
        None, "--format", "-f", help="Output format [jsonl|alpaca|ft|chatml]"
    ),
    storage: str = typer.Option(
        "json", "--storage", help="Storage format [json|hf]",
        show_default=True
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
):
    """
    Convert to different formats for fine-tuning.
    
    The --format option controls the content format (how the data is structured).
    The --storage option controls how the data is stored (JSON file or HF dataset).
    
    When using --storage hf, the output will be a directory containing a Hugging Face 
    dataset in Arrow format, which is optimized for machine learning workflows.
    """
    from synthetic_data_kit.core.save_as import convert_format
    
    # Get format from args or config
    if not format:
        format_config = ctx.config.get("format", {})
        format = format_config.get("default", "jsonl")
    
    # Set default output path if not provided
    if not output:
        final_dir = get_path_config(ctx.config, "output", "final")
        os.makedirs(final_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input))[0]
        
        if storage == "hf":
            # For HF datasets, use a directory name
            output = os.path.join(final_dir, f"{base_name}_{format}_hf")
        else:
            # For JSON files, use appropriate extension
            if format == "jsonl":
                output = os.path.join(final_dir, f"{base_name}.jsonl")
            else:
                output = os.path.join(final_dir, f"{base_name}_{format}.json")
    
    try:
        with console.status(f"Converting {input} to {format} format with {storage} storage..."):
            output_path = convert_format(
                input,
                output,
                format,
                ctx.config,
                storage_format=storage
            )
        
        if storage == "hf":
            console.print(f" Converted to {format} format and saved as HF dataset to [bold]{output_path}[/bold]", style="green")
        else:
            console.print(f" Converted to {format} format and saved to [bold]{output_path}[/bold]", style="green")
        return 0
    except Exception as e:
        console.print(f"L Error: {e}", style="red")
        return 1


@app.command()
def translate(
    input: str = typer.Argument(..., help="JSON file to translate"),
    lang: str = typer.Option(..., "--lang", "-l", help="Target language (e.g., 'french', 'spanish')"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Where to save the output"
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base", help="VLLM API base URL"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="VLLM API key"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
):
    """
    Translate JSON content to target language using LLM.
    
    The input file must be a valid JSON file. The translation will preserve the JSON structure
    while translating all string values to the target language.
    """
    from synthetic_data_kit.core.translate import process_file
    
    # Get VLLM server details from args or config
    vllm_config = get_vllm_config(ctx.config)
    api_base = api_base or vllm_config.get("api_base")
    api_key = api_key or vllm_config.get("api_key")
    model = model or vllm_config.get("model")
    
    # Check server first
    if check_server(api_base, api_key, model, console) != 0:
        return
    
    # Get output directory from args, then config, then default
    if output_dir is None:
        output_dir = get_path_config(ctx.config, "output", "translated")
    
    try:
        with console.status(f"Translating {input} to {lang}..."):
            output_path = process_file(
                input,
                output_dir,
                lang,
                ctx.config_path,
                api_base,
                model,
                verbose
            )
        console.print(f" Translation saved to [bold]{output_path}[/bold]", style="green")
        return 0
    except json.JSONDecodeError as e:
        console.print(f"L Error: Input file is not valid JSON: {e}", style="red")
        return 1
    except Exception as e:
        console.print(f"L Error: {e}", style="red")
        return 1



# ======================= Pipeline 1 Steps: Cleaning + Citations + Reasoning # ======================= #

@app.command("cite")
def add_citations(
    input: str = typer.Argument(..., help="Path to JSON file with QA pairs"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Where to save the citation-enhanced output"
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base", help="VLLM API base URL"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="VLLM API key"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use"
    ),
    batch_size: int = typer.Option(
        8, "--batch-size", "-b", help="Number of QA pairs per batch"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
):
    """
    Add citations to QA pairs using a language model.

    This command reads a JSON file containing a list of QA pairs under the 'qa_pairs' key,
    prompts the LLM to enhance them with citations, and saves the result.
    """
    from synthetic_data_kit.core.add_citations import add_citations_to_qa_pairs
    
    # Get VLLM server details from args or config
    vllm_config = get_vllm_config(ctx.config)
    api_base = api_base or vllm_config.get("api_base")
    api_key = api_key or vllm_config.get("api_key")
    model = model or vllm_config.get("model")
    
    # Check server first    
    if check_server(api_base, api_key, model, console) != 0:
        return
    
    if output_dir is None:
        output_dir = get_path_config(ctx.config, "output", "add_citations")

    try:
        with console.status(f"Adding citations to QA pairs from {input}..."):
            output_path = add_citations_to_qa_pairs(
                file_path=input,
                config_path=ctx.config_path,
                api_base=api_base,
                model=model,
                output_dir=str(output_dir),
                batch_size=batch_size,
                verbose=verbose
            )
        if output_path:
            console.print(f" Citations added and saved to [bold]{output_path}[/bold]", style="green")
        return 0
    except Exception as e:
        console.print(f"L Error: {e}", style="red")
        return 1

@app.command("reason-cite")
def add_citations_with_reasoning(
    input: str = typer.Argument(..., help="Path to JSON file with QA pairs"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Where to save the citation-enhanced output with reasoning"
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base", help="VLLM API base URL"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="VLLM API key"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use"
    ),
    batch_size: int = typer.Option(
        8, "--batch-size", "-b", help="Number of QA pairs per batch"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
):
    """
    Add citations with detailed reasoning to QA pairs.
    
    This command processes QA pairs and adds:
    1. Citations from the source text
    2. Detailed reasoning explaining why each citation was chosen
    3. Step-by-step thought process for citation selection
    """
    from synthetic_data_kit.core.add_citations_with_reasoning import process_file as process_citations
    
    # Get VLLM server details from args or config
    vllm_config = get_vllm_config(ctx.config)
    api_base = api_base or vllm_config.get("api_base")
    api_key = api_key or vllm_config.get("api_key")
    model = model or vllm_config.get("model")
    
    # Check server first
    if check_server(api_base, api_key, model, console) != 0:
        return 1
    
    # Get output directory from args, then config, then default
    if output_dir is None:
        output_dir = get_path_config(ctx.config, "output", "cited_with_reasoning")
    
    try:
        with console.status(f"Adding citations with reasoning to QA pairs from {input}..."):
            output_path = process_citations(
                input,
                config_path=ctx.config_path,
                api_base=api_base,
                model=model,
                output_dir=output_dir,
                batch_size=batch_size,
                verbose=verbose
            )
        console.print(f" Citations and reasoning added, saved to [bold]{output_path}[/bold]", style="green")
        return 0
    except Exception as e:
        console.print(f"L Error: {str(e)}", style="red")
        return 1

@app.command("curate-w-reason")
def curate_w_reason(
    input: str = typer.Argument(..., help="Input file to clean"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    threshold: Optional[float] = typer.Option(
        None, "--threshold", "-t", help="Quality threshold (1-10)"
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base", help="VLLM API base URL"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="VLLM API key"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
):
    """
    Clean and filter content based on quality.
    """
    from synthetic_data_kit.core.curate_w_reason import process_file
    
    # Get VLLM server details from args or config
    vllm_config = get_vllm_config(ctx.config)
    api_base = api_base or vllm_config.get("api_base")
    api_key = api_key or vllm_config.get("api_key")
    model = model or vllm_config.get("model")
    
    # Check server first
    if check_server(api_base, api_key, model, console) != 0:
        return
    
    try:
        with console.status(f"Cleaning content from {input}..."):
            result_path = process_file(
                file_path=input,
                output_dir= "data/cleaned" if output == None else output,
                threshold=threshold,
                api_base=api_base,
                model=model,
                config_path=ctx.config_path,
                verbose=verbose,
                action="clean_and_filter"
            )
        console.print(f" Cleaned content saved to [bold]{result_path}[/bold]", style="green")
        return 0
    except Exception as e:
        console.print(f"L Error: {e}", style="red")
        return 1

# ---------------------------------

@app.command("run-pipe")
def run_pipeline(
    input: str = typer.Argument(..., help="Directory of input files to process"),
    content_type: str = typer.Option(
        "qa", "--type", help="Type of content to generate [qa|summary|cot|cot-enhance]"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Where to save the output"
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base", help="VLLM API base URL"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="VLLM API key"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use"
    ),
    num_files: Optional[int] = typer.Option(
        None, "--num-files", "-n", help="Target number of input files to process"
    ),
    num_pairs: Optional[int] = typer.Option(
        5, "--num-pairs", "-p", help="Target number of QA pairs to generate per file"
    ),
    pipeline: str = typer.Option(
        "v1", "--pipeline", "-pl", help="Pipeline version to use [v1|v2]"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
):
    from synthetic_data_kit.models.llm_client import LLMClient
    from synthetic_data_kit.core.run_pipeline import process_file as pipeline_v1
    from synthetic_data_kit.generators.qa_generator_detailed import QAGenerator
    from synthetic_data_kit.pipeline.run_pipeline_v2 import process_file as pipeline_v2

    # Get VLLM server details from args or config
    vllm_config = get_vllm_config(ctx.config)
    api_base = api_base or vllm_config.get("api_base")
    api_key = api_key or vllm_config.get("api_key")
    model = model or vllm_config.get("model")

    # Check server first
    if check_server(api_base, api_key, model, console) != 0:
        return

    # Get output directory from args, then config, then default
    if output_dir is None:
        output_dir = get_path_config(ctx.config, "output", "generated")

    try:
        with console.status(f"Running [bold]{pipeline}[/bold] pipeline for generating [bold]{content_type}[/bold] content from [bold]{input}[/bold]..."):
            if pipeline == "v1":
                output_path = pipeline_v1(
                    input_dir=input,
                    output_dir=output_dir,
                    config_path=ctx.config_path,
                    api_base=api_base,
                    model=model,
                    num_files=num_files if num_files is not None else 10,
                )
            elif pipeline == "v2":
                
                client = LLMClient(config_path=ctx.config_path, api_base=api_base, model_name=model)
                config = client.config
                generator = QAGenerator(client=client, config_path=ctx.config_path)
                
                output_path = pipeline_v2(
                    input_dir=input,
                    output_dir=output_dir,
                    num_files=num_files if num_files is not None else 10,
                    config=config,
                    client=client,
                    generator=generator,
                    num_pairs=num_pairs,
                    verbose=verbose
                )
            else:
                console.print(f"[red]Error:[/red] Unknown pipeline version: {pipeline}")
                return 1

        if output_path:
            console.print(f"\n✅ [bold]{pipeline.upper()} pipeline completed![/bold] Content saved to [green]{output_path}[/green]")
        return 0
    except Exception as e:
        console.print(f"[red]❌ Error:[/red] {e}")
        return 1



# ======================================= Helper Commands ========================================== #

# To See Stats of the Generated qa pairs 
@app.command("stats")
def stats(input: str = typer.Argument(..., help="Directory of input files to process")):

    # Step 1: Save the Locations of all the JSON files
    json_files = [f for f in os.listdir(input) if f.endswith('.json') and os.path.isfile(os.path.join(input, f))]

    total_items = 0
    file_count = 0

    for filename in json_files:
        file_path = os.path.join(input, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                qa_pairs = data.get("qa_pairs", [])
                if isinstance(qa_pairs, list):
                    total_items += len(qa_pairs)
                    file_count += 1
                else:
                    typer.echo(f"Warning: 'qa_pairs' in {filename} is not a list.")
        except Exception as e:
            typer.echo(f"Error reading {filename}: {e}")
    
    if file_count == 0:
        typer.echo("No valid JSON files with 'qa_pairs' found.")
        return

    average = total_items / file_count

    # Step 4: Output the stats
    console.print(f"Processed JSON files: {file_count} ")
    console.print(f"Total QA pairs: {total_items}")
    console.print(f"Average QA pairs per file: {average:.2f}")


# ---------------------------------- PROCESSING AGEFI DATA ---------------------------------->>>>>>>>>>

# To Reformat JSONL formats into appropriate JSONL format for ingestion
@app.command("reformat-jsonl")
def reformat(input: str = typer.Argument(..., help="Input file to reformat")):
    input_path = Path(input)
    result_path = input_path.parent / f"{input_path.stem}_reformatted.jsonl"

    try:
        with console.status(f"Reformatting content from {input_path}..."):
            with input_path.open("r", encoding="utf-8") as infile, result_path.open("w", encoding="utf-8") as outfile:
                for line in infile:
                    obj = json.loads(line)
                    new_obj = {
                        "title": obj.get("headline", ""),
                        "text": obj.get("articleBodyRendered", "")
                    }
                    outfile.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

        console.print(f"Cleaned content saved to [bold]{result_path}[/bold]", style="green")
        return 0
    
    except Exception as e:
        console.print(f"Error: {e}", style="red")
        return 1

# To extract QA pairs for further processing from a JSONL file
@app.command("extract-qa")
def extract_qa(
    input: str = typer.Argument(..., help="Input JSONL file to extract QA pairs from"),
    batch_size: int = typer.Option(25, help="Number of QA pairs per output file")
):
    
    def add_sentence_markers(text):
        # Split the text into sentences using regex (handles '.', '!', '?')
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        # Add sentence markers
        marked_sentences = [f"[S{i+1}]{sentence}" for i, sentence in enumerate(sentences) if sentence]
        
        # Join the marked sentences into a final string
        return ''.join(marked_sentences)
    
    def split_into_batches(data, batch_size):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    
    input_path = Path(input)
    input_filename = input_path.stem
    output_dir = Path("data") / "generated" / input_filename
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        qa_pairs = []

        with console.status(f"Reading from {input_path}..."):
            with input_path.open("r", encoding="utf-8") as infile:
                for line in infile:
                    obj = json.loads(line)
                    chunk = obj.get("articleBodyRendered")
                    question = obj.get("question")
                    answer = obj.get("answer")

                    if chunk is not None and question is not None and answer is not None:
                        qa_pairs.append({
                            "chunk": add_sentence_markers(chunk),
                            "question": question,
                            "answer": answer
                        })

        if not qa_pairs:
            console.print(f"No valid QA pairs found in the file.", style="yellow")
            return 1

        batches = split_into_batches(qa_pairs, batch_size)

        with console.status(f"Saving {len(batches)} batch(es) to {output_dir}..."):
            for i, batch in enumerate(batches):
                output_data = {"qa_pairs": batch}
                output_file = output_dir / f"qa_batch_{i+1}.json"
                with output_file.open("w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)

        console.print(f"QA pairs saved to [bold]{output_dir.resolve()}[/bold]", style="green")
        return 0

    except FileNotFoundError:
        console.print(f"Error: Input file '{input}' not found.", style="red")
        return 1
    except json.JSONDecodeError as e:
        console.print(f"Error decoding JSON: {e}", style="red")
        return 1
    except Exception as e:
        console.print(f"Unexpected error: {e}", style="red")
        return 1

# ------------------------------------------------------------------------------------------>>>>>>>>>>

@app.command("clean-chunks")
def clean_chunks(
    input_dir: str = typer.Argument(..., help="Directory containing JSON files with QA pairs"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Output directory path"
    ),
):
    """
    Clean chunks in QA pairs by removing 'Chunk:\n\n' prefix from all JSON files in a directory.
    """
    try:
        input_path = Path(input_dir)
        if not input_path.is_dir():
            console.print(f"Error: '{input_dir}' is not a directory", style="red")
            return 1

        # Setup output directory
        if output_dir is None:
            output_dir = input_path.parent / f"{input_path.name}"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all JSON files
        json_files = list(input_path.glob("*.json"))
        if not json_files:
            console.print(f"No JSON files found in {input_dir}", style="yellow")
            return 1

        processed_files = 0
        total_qa_pairs = 0
        cleaned_chunks = 0

        # Process each JSON file
        with console.status(f"Processing {len(json_files)} JSON files...") as status:
            for json_file in json_files:
                try:
                    # Read input file
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if "qa_pairs" not in data:
                        console.print(f"Skipping {json_file.name}: No 'qa_pairs' found", style="yellow")
                        continue

                    # Clean chunks
                    file_cleaned_chunks = 0
                    for qa_pair in data["qa_pairs"]:
                        if "chunk" in qa_pair:
                            chunk = qa_pair["chunk"]
                            if chunk.startswith("Chunk:\n\n"):
                                qa_pair["chunk"] = chunk[len("Chunk:\n\n"):]
                                file_cleaned_chunks += 1

                    # Save cleaned data
                    output_file = output_dir / json_file.name
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

                    processed_files += 1
                    total_qa_pairs += len(data["qa_pairs"])
                    cleaned_chunks += file_cleaned_chunks
                    
                    status.update(f"Processed {processed_files}/{len(json_files)} files...")

                except json.JSONDecodeError as e:
                    console.print(f"Error in {json_file.name}: Invalid JSON: {e}", style="red")
                except Exception as e:
                    console.print(f"Error processing {json_file.name}: {str(e)}", style="red")

        # Print summary
        console.print("\nSummary:", style="bold blue")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Files Processed", str(processed_files))
        table.add_row("Total QA Pairs", str(total_qa_pairs))
        table.add_row("Chunks Cleaned", str(cleaned_chunks))
        table.add_row("Output Directory", str(output_dir))
        console.print(table)

        return 0

    except Exception as e:
        console.print(f"Error: {str(e)}", style="red")
        return 1


@app.command("filter-citations")
def filter_citations_cmd(
    input: str = typer.Argument(..., help="Path to JSON file with cited QA pairs"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Where to save the filtered output"
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base", help="VLLM API base URL"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="VLLM API key"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use"
    ),
    verbose: bool = typer.Option(
        True, "--verbose", "-v", help="Show detailed output"
    ),
):
    """
    Filter QA pairs based on citation validity.
    
    This command processes QA pairs and:
    1. Checks if citations properly support the answers
    2. Verifies citation format and relevance
    3. Filters out pairs with low-quality citations
    """
    from synthetic_data_kit.core.filter_citations import process_file as filter_citations
    
    # Get VLLM server details from args or config
    vllm_config = get_vllm_config(ctx.config)
    api_base = api_base or vllm_config.get("api_base")
    api_key = api_key or vllm_config.get("api_key")
    model = model or vllm_config.get("model")
    
    # Check server first
    if check_server(api_base, api_key, model, console) != 0:
        return 1
    
    # Get output directory from args, then config, then default
    if output_dir is None:
        output_dir = get_path_config(ctx.config, "output", "reasoning_distil")
    
    try:
        with console.status(f"Filtering citations in QA pairs from {input}..."):
            output_path = filter_citations(
                input,
                config_path=ctx.config_path,
                api_base=api_base,
                model=model,
                output_dir=output_dir,
                verbose=verbose
            )
        console.print(f"Citations filtered, saved to [bold]{output_path}[/bold]", style="green")
        return 0
    except Exception as e:
        console.print(f"Error: {str(e)}", style="red")
        return 1


# ======================================= Evaluation Commands ========================================== #




@app.command("finetune")
def finetune(
    device: str = typer.Option(
        '1', "--device", "-d", help="GPU Device to be used"
    ),
):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(device))
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)

    # Hardcoded list of directories to fine-tune on
    input_dirs = [
        # 'data/generated/en-wikipedia-finance.txt',  # v1.0
        # 'data/generated/en-wikipedia-finance.txt_reasoning_distilled',  # v2.1
        'data/generated/en-wikipedia-finance.txt_pipeline_v2'  # v2.2
    ]

    # Configurations
    models = [
        "Qwen/Qwen2.5-3B-Instruct",
        # "Qwen/Qwen2.5-7B-Instruct"
    ]
    sample_sizes = [
        # 500, 
        1000
    ]

    for model in models:
        for num_of_samples in sample_sizes:
            for input_dir in input_dirs:
                
                from synthetic_data_kit.evaluation.finetune import training
                console.print(f"[yellow]Before finetune: Free GPU memory: {info.free // (1024 ** 2)} MiB")
                try:
                    # Derive base name of the input file
                    filename = os.path.basename(input_dir)
                    if ".txt" in filename:
                        dataset_name = filename.split(".txt", 1)[0] + filename.split(".txt", 1)[1]
                    else:
                        dataset_name = os.path.splitext(filename)[0]

                    # Normalize model name
                    model_base = model.split("/")[-1].lower().replace("-instruct", "").replace("-chat", "").replace("_", "-")

                    # Construct finetune model name
                    ft_model_name = f"{model_base}-{dataset_name}-{num_of_samples}-vtest"
                    
                    console.rule(f"[bold blue]Finetuning Model: {ft_model_name}[/bold blue]")
                    
                    # Perform training
                    
                    hf_loc = training(
                        num_of_samples=num_of_samples,
                        model_name=model,
                        ft_model_name=ft_model_name,
                        input_dir=input_dir,
                    )

                    print(f"Finetuned Model for {input_dir}: {hf_loc}")

                    # Clear GPU memory from the previous run
                    torch.cuda.empty_cache()
                except Exception as e:
                    console.print(f"[red]Error during finetuning {ft_model_name}: {e}[/red]")
                finally:
                    del training
                    torch.cuda.empty_cache()
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    console.print(f"[green]After finetune: Free GPU memory: {info.free // (1024 ** 2)} MiB")
        
@app.command("evaluate")
def evaluate_all_models(
    split: str = typer.Option(
        "test", "--split", "-s", help="Dataset split to evaluate on"
    ),
    device: str = typer.Option(
        '1', "--device", "-d", help="GPU Device to be used"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
):
    """
    Evaluate all fine-tuned models on multiple question answering datasets.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    from synthetic_data_kit.evaluation.evaluation import ModelEvaluator
    import torch
    import gc
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(device))
    

    datasets = [
                "2wikimqa_e", 
                # "multifieldqa_en",
                # "musique",
                # "narrativeqa"
            ]

    
    # system prompt 2
    # FINE_TUNED_MODELS = [
    #     "Qwen/qwen2.5-7b-Instruct", # Base Model
    #     "ibrahimbukhariLingua/qwen2.5-3b-en-wikipedia-finance_pipeline_v2-500-v3", # v2.2
        
    # ] 
    
    # system prompt 1
    FINE_TUNED_MODELS = [
        # "ibrahimbukhariLingua/qwen2.5-3b-en-wikipedia-finance_reasoning_distilled-500-v1", # data-v2.1
        # "ibrahimbukhariLingua/qwen2.5-3b-en-wikipedia-finance_pipeline_v2-500-v1", # data-v2.2
        "ibrahimbukhariLingua/qwen2.5-7b-en-wikipedia-finance-1000-v4", # data-v1
        "ibrahimbukhariLingua/qwen2.5-7b-en-wikipedia-finance-500-v4", # data-v1
        "ibrahimbukhariLingua/qwen2.5-3b-en-wikipedia-finance-1000-v4", # data-v1
        "ibrahimbukhariLingua/qwen2.5-3b-en-wikipedia-finance-500-v4", # data-v1
    ] 

    

    for model in FINE_TUNED_MODELS:
        # Report GPU memory before evaluation
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        console.print(f"[yellow]Before evaluation: Free GPU memory: {info.free // (1024 ** 2)} MiB")

        console.rule(f"[bold blue]Evaluating Model: {model}")
        evaluator = ModelEvaluator(
                model_name=model,
                verbose=verbose
            )
        try:
            
            main_table = Table(title=f"Evaluation Results: {model}")
            main_table.add_column("Dataset", style="cyan")
            main_table.add_column("Samples", style="magenta")
            main_table.add_column("Accuracy", style="green")
            main_table.add_column("Results File", style="blue")

            total_samples = 0
            total_correct = 0

            for dataset_name in datasets:
                with console.status(f"Evaluating {model} on {dataset_name} {split} split..."):

                    result = evaluator.evaluate_dataset(dataset_name, split)

                    samples = result["samples"]
                    accuracy = result["metrics"]["accuracy"]
                    total_samples += samples
                    total_correct += int(accuracy * samples)

                    main_table.add_row(
                        dataset_name,
                        str(samples),
                        f"{accuracy:.2%}",
                        result["output_file"]
                    )

            overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
            main_table.add_row(
                "Overall",
                str(total_samples),
                f"{overall_accuracy:.2%}",
                "---",
                style="bold"
            )

            console.print(main_table)
            console.print("\n")

        except Exception as e:
            console.print(f"[red]Error evaluating model {model}: {str(e)}")

        finally:
            # Explicitly delete evaluator and clear memory
            del evaluator
            gc.collect()
            torch.cuda.empty_cache()

            # Log GPU memory after cleanup
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            console.print(f"[green]After cleanup: Free GPU memory: {info.free // (1024 ** 2)} MiB")

@app.command("results")
def results(
    input_dir: str = typer.Argument(..., help="Directory containing JSON files with QA pairs"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug output")
):
    """
    Reads each JSONL file in the given directory, extracts the "accuracy" value from the "metrics" field,
    and prints the results. Each JSONL file represents a benchmark test, and each directory represents a model.
    """
    
    console.print(f"[bold]Reading results from directory: {input_dir}[/bold]")
    
    for filename in os.listdir(input_dir):
        if not filename.endswith(".jsonl"):
            continue

        benchmark_name = filename.replace(".jsonl", "")
        file_path = os.path.join(input_dir, filename)

        if verbose:
            console.print(f"[DEBUG] Processing file: {file_path}")

        try:
            with open(file_path, "r") as f:
                for line_num, line in enumerate(f, start=1):
                    data = json.loads(line)
                    accuracy = data.get("metrics", {}).get("accuracy")
                    if accuracy is not None:
                        console.print(f"{benchmark_name}: {accuracy}")
                        if verbose:
                            console.print(f"[DEBUG] Line {line_num}: accuracy = {accuracy}")
                    else:
                        console.print(f"{benchmark_name}: 'accuracy' key not found")
                        if verbose:
                            console.print(f"[DEBUG] Line {line_num}: Missing 'accuracy'")
        except Exception as e:
            console.print(f"[red]Error reading {file_path}: {e}[/red]")
            if verbose:
                console.print(f"[DEBUG] Exception encountered: {e}")

@app.command("compare")
def compare(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug output"),
    max_mismatches: int = typer.Option(2, "--max", "-m", help="Maximum mismatches to show")
):
    """
    Compares two JSONL files and prints up to `max_mismatches` mismatches where labels differ.
    Skips mismatches where either generated answer is empty.
    Displays responses side-by-side with their model names and labels.
    """

    # Define paths and model names
    file1_path = "data/results/ibrahimbukhariLingua_qwen2.5-7b-en-wikipedia-finance_reasoning_distilled-1000-v2/2wikimqa_e.jsonl"
    file2_path = "data/results/ibrahimbukhariLingua_qwen2.5-7b-en-wikipedia-finance_reasoning_distilled-1000-v3/2wikimqa_e.jsonl"
    model1_name = "qwen2.5-7b-v2"
    model2_name = "qwen2.5-7b-v3"

    console.print(f"[bold]Comparing files:[/bold]\n- {file1_path} ({model1_name})\n- {file2_path} ({model2_name})")

    try:
        # Load JSONL files
        with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
            lines1 = [json.loads(line.strip()) for line in f1 if line.strip()]
            lines2 = [json.loads(line.strip()) for line in f2 if line.strip()]

        mismatches = []

        # Compare entries line by line
        for idx, (entry1, entry2) in enumerate(zip(lines1, lines2), start=1):
            responses1 = entry1.get("all_responses", [])
            responses2 = entry2.get("all_responses", [])

            # Compare individual responses
            for i, (resp1, resp2) in enumerate(zip(responses1, responses2)):
                label1 = resp1.get("label")
                label2 = resp2.get("label")
                answer1 = resp1.get("generated_answer", "").strip()
                answer2 = resp2.get("generated_answer", "").strip()

                # Skip if either answer is empty
                if not answer1 or not answer2:
                    if verbose:
                        console.print(f"[DEBUG] Skipped empty answer at entry {idx}, response {i}")
                    continue

                if label1 != label2:
                    # Alternate display order
                    if len(mismatches) % 2 == 0:
                        mismatches.append((model1_name, answer1, label1, model2_name, answer2, label2))
                    else:
                        mismatches.append((model2_name, answer2, label2, model1_name, answer1, label1))

                    if verbose:
                        console.print(
                            f"[DEBUG] Mismatch at entry {idx}, response {i}:\n"
                            f"    {model1_name} label = {label1}\n"
                            f"    {model2_name} label = {label2}"
                        )

                    if len(mismatches) >= max_mismatches:
                        break

            if len(mismatches) >= max_mismatches:
                break

        # Display results
        if mismatches:
            console.print(f"[bold green]Found mismatches (showing up to {max_mismatches}):[/bold green]\n")
            for i, (name1, answer1, label1, name2, answer2, label2) in enumerate(mismatches, start=1):
                console.print(f"[{i}]")
                console.print(f"[yellow]{name1} (label={label1}):[/yellow] {answer1}")
                console.print(f"[magenta]{name2} (label={label2}):[/magenta] {answer2}\n")
        else:
            console.print("[yellow]No mismatches found.[/yellow]")

    except Exception as e:
        # Handle errors
        console.print(f"[red]Error comparing files: {e}[/red]")
        if verbose:
            console.print(f"[DEBUG] Exception encountered: {e}")
            

@app.command("run-eval")
def runEval(
    version: str = typer.Option(
        None, "--version", "-ver", help="Finetuning Version" 
    ),
    device: int = typer.Option(
        0, "--device", "-d", help="GPU Device to be used"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
):
    # setting device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    
    from synthetic_data_kit.evaluation.run_eval import run_finetuning_all_configs, run_evaluation_from_models_folder

    def display_results_table(all_results):
        for repo_result in all_results:
            for repo_id, benchmarks in repo_result.items():
                main_table = Table(title=f"Evaluation Results: {repo_id}")
                main_table.add_column("Dataset", style="cyan")
                main_table.add_column("Samples", style="magenta")
                main_table.add_column("Accuracy", style="green")
                main_table.add_column("Results File", style="blue")

                for benchmark_result in benchmarks:
                    dataset = benchmark_result.get("dataset", "N/A")
                    samples = str(benchmark_result.get("samples", "N/A"))
                    accuracy = str(benchmark_result.get("metrics", {}).get("accuracy", "N/A"))
                    output_file = benchmark_result.get("output_file", "N/A")

                    main_table.add_row(dataset, samples, accuracy, output_file)

                console.print(main_table)


    # Hardcoded list of directories to fine-tune on
    input_dirs = [
        # 'data/generated/en-wikipedia-finance.txt',  # v1.1
        'data/generated/en-wikipedia-finance.txt_reasoning_distilled',  # v2.1
        'data/generated/en-wikipedia-finance.txt_pipeline_v2'  # v2.2
    ]
    console.print(f"Input directories: {input_dirs}")
    
    # Benchmarks to evaluate
    benchmarks = [
        "2wikimqa_e", 
        # "multifieldqa_en",
        # "musique",
        # "narrativeqa"
    ]
    console.print(f"Benchmarks to evaluate: {benchmarks}")
    
    
    # Configurations
    models = [
        # "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct"
    ]
    console.print(f"Models to fine-tune: {models}")

    run_finetuning_all_configs(
        rag_index_dir="data/output/en-wikipedia-finance.txt",
        dataset_dirs=input_dirs,
        model_variants=models,
        version=version,
        verbose=verbose)
    
    results = run_evaluation_from_models_folder(
        benchmarks=benchmarks,
        version=version,
        verbose=verbose)
    
    display_results_table(results)






# ======================= Pipeline 2 Steps: QA Generation # ======================= #

@app.command("create-qa-v2")
def generateQA_v2(
    input: str = typer.Argument(..., help="Path to the processed text file"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Where to save the output"
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base", help="VLLM API base URL"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="VLLM API key"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use"
    ),
    num_pairs: Optional[int] = typer.Option(
        5, "--num-pairs", "-n", help="Target number of QA pairs to generate"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
):
    """
    Generate detailed QA pairs from a processed text file using an enhanced generation pipeline.
    
    This command uses a more sophisticated QA generation approach that:
    1. Splits documents into optimal chunks
    2. Adds sentence markers for better context
    3. Generates a document summary first
    4. Creates detailed QA pairs with citations
    """
    from synthetic_data_kit.generators.qa_generator_detailed import QAGenerator
    from synthetic_data_kit.models.llm_client import LLMClient
    
    # Get VLLM server details from args or config
    vllm_config = get_vllm_config(ctx.config)
    api_base = api_base or vllm_config.get("api_base")
    api_key = api_key or vllm_config.get("api_key")
    model = model or vllm_config.get("model")
    
    # Check server first
    if check_server(api_base, api_key, model, console) != 0:
        return 1
    
    # Get output directory from args, then config, then default
    if output_dir is None:
        output_dir = get_path_config(ctx.config, "output", "generated")

    if verbose:
        # No try/except: let errors raise for debugging
        # Read the input text file directly
        with console.status(f"Reading input file {input}..."):
            with open(input, 'r', encoding='utf-8') as f:
                document_text = f.read()

        # Initialize the QA generator with correct parameter names
        llm_client = LLMClient(
            api_base=api_base,
            model_name=model,
            api_key=api_key,
            config_path=ctx.config_path
        )
        generator = QAGenerator(client=llm_client, config_path=ctx.config_path)

        # Generate QA pairs
        with console.status(f"Generating QA pairs from text..."):
            output_path = generator.process_document(
                input_file_path=input,
                num_pairs=num_pairs,
                output_dir=str(output_dir),
                verbose=verbose
            )

        console.print(f" QA pairs generated and saved to [bold]{output_path}[/bold]", style="green")
        return 0
    else:
        try:
            # Read the input text file directly
            with console.status(f"Reading input file {input}..."):
                try:
                    with open(input, 'r', encoding='utf-8') as f:
                        document_text = f.read()
                except FileNotFoundError:
                    console.print(f"L Error: Input file '{input}' not found", style="red")
                    return 1
                except Exception as e:
                    console.print(f"L Error reading input file: {e}", style="red")
                    return 1

            # Initialize the QA generator with correct parameter names
            llm_client = LLMClient(
                api_base=api_base,
                model_name=model,
                api_key=api_key,
                config_path=ctx.config_path
            )
            generator = QAGenerator(client=llm_client, config_path=ctx.config_path)

            # Generate QA pairs
            with console.status(f"Generating QA pairs from text..."):
                output_path = generator.process_document(
                    input_file_path=input,
                    num_pairs=num_pairs,
                    output_dir=str(output_dir),
                    verbose=verbose
                )

            console.print(f" QA pairs generated and saved to [bold]{output_path}[/bold]", style="green")
            return 0

        except Exception as e:
            console.print(f"L Error: {e}", style="red")
            return 1

@app.command("preprocess")
def pre_proccess_citations(
    input: str = typer.Argument(..., help="Path to the processed text file"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Where to save the output"
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base", help="VLLM API base URL"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),):
    
    from synthetic_data_kit.models.llm_client import LLMClient
    from synthetic_data_kit.pipeline.pre_process_citations import process_file
    
    # Get VLLM server details from args or config
    vllm_config = get_vllm_config(ctx.config)
    api_base = api_base or vllm_config.get("api_base")
    api_key = vllm_config.get("api_key")
    model = model or vllm_config.get("model")
    
    # Check server first
    if check_server(api_base, api_key, model, console) != 0:
        return 1
    
    # Get output directory from args, then config, then default
    if output_dir is None:
        output_dir = get_path_config(ctx.config, "output", "preprocess")
    
    client = LLMClient(config_path=ctx.config_path, api_base=api_base, model_name=model)
    config = client.config
    
    try:
        with console.status(f"Cleaning and Adding Reasoning + Ciations on {input}..."):
            
            output_path = process_file (file_path = input, output_dir = output_dir, client=client, config=config, verbose = verbose)
            
        console.print(f" Citations and reasoning added, saved to [bold]{output_path}[/bold]", style="green")
        return 0
    except Exception as e:
        console.print(f"L Error: {str(e)}", style="red")
        return 1

@app.command("postprocess")
def post_proccess_citations(
    input: str = typer.Argument(..., help="Path to the processed text file"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Where to save the output"
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base", help="VLLM API base URL"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),):
    
    from synthetic_data_kit.models.llm_client import LLMClient
    from synthetic_data_kit.pipeline.post_process_citations import process_file
    
    # Get VLLM server details from args or config
    vllm_config = get_vllm_config(ctx.config)
    api_base = api_base or vllm_config.get("api_base")
    api_key = vllm_config.get("api_key")
    model = model or vllm_config.get("model")
    
    # Check server first
    if check_server(api_base, api_key, model, console) != 0:
        return 1
    
    # Get output directory from args, then config, then default
    if output_dir is None:
        output_dir = get_path_config(ctx.config, "output", "postprocess")
    
    client = LLMClient(config_path=ctx.config_path, api_base=api_base, model_name=model)
    config = client.config
    
    try:
        with console.status(f"Filtering and Combining Ciations from {input}..."):
            
            output_path = process_file (file_path = input, output_dir = output_dir, client=client, config=config, verbose = verbose)
            
        console.print(f" filtered_and_combined, saved to [bold]{output_path}[/bold]", style="green")
        return 0
    except Exception as e:
        console.print(f"L Error: {str(e)}", style="red")
        return 1
    
    


if __name__ == "__main__":
    app()
