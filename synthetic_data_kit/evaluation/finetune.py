# Basic imports
import os, json, re
import numpy as np
from tqdm import tqdm

# PyTorch and HuggingFace imports
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from huggingface_hub.hf_api import HfFolder

# Custom imports
from synthetic_data_kit.evaluation.util import transform_input, RAGFormatter

# Set the environment variable for HuggingFace token
HfFolder.save_token("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")



# ======================= Data Processing ========================== #

def preprocess_example(example: dict, formatter:RAGFormatter) -> dict:
    """
    Processes a single example using a RAGFormatter.

    Args:
        example (dict): A dictionary with at least 'chunk', 'question', and 'answer'.
                        Optionally includes 'thinking' or 'reasoning'.
        formatter (RAGFormatter): An initialized RAGFormatter object.

    Returns:
        dict: A dictionary with formatted passages, updated answer, and updated thinking.
    """
    # Step 1: Ensure chunk_list is available
    gold_passages = example.get('chunk_list')
    if not gold_passages:
        gold_passages = [example['chunk']]

    # Step 2: Extract gold thinking
    gold_thinking = example.get('reasoning') or example.get('thinking', '')

    # Step 3: Format using RAGFormatter
    formatted_passages, updated_answer, updated_thinking = formatter.format_for_question(
        question=example['question'],
        gold_passages=gold_passages,
        gold_answer=example['answer'],
        gold_thinking=gold_thinking
    )

    # Step 4: Return processed example
    return {
        'chunk': formatted_passages,
        'answer': transform_input(updated_answer),
        'thinking': updated_thinking,
        'question': example['question']
    }

def apply_template(example):
    # ---------------------------------------- System Prompt ----------------------------------------
    system_prompt = (
        "You are a question answering system which will be provided with a Source passage. "
        "Using the Source passage you will answer the user's question. "
        "You will cite sources which you use when answering the question."
    )

    # ---------------------------------------- Format Source Passages ----------------------------------------
    formatted_passages = "\n".join(
        f"<Passage{i+1}> {p} </Passage{i+1}>" for i, p in enumerate(example['chunk'])
    )

    # ---------------------------------------- User Prompt ----------------------------------------
    prompt = (
        f"Source Passages:\n{formatted_passages}\n\n"
        f"User Question:\n{example['question']}"
    )

    # ---------------------------------------- Assistant Completion ----------------------------------------
    completion = f"<think>{example['thinking']}</think>\n{example['answer']}"
    # completion = example['answer']

    # ---------------------------------------- Message Format ----------------------------------------
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': completion}
    ]

    return {'messages': messages}

def process_dataset(directory_path: str, num_of_samples: int, formatter:RAGFormatter) -> Dataset:
    all_qa_pairs = []
    
    # Function to extract QA pairs from a JSON file
    def extract_qa_pairs(json_file_path:str):
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data.get("qa_pairs")
    
    # Step 1: Extract all JSON file paths in the directory  
    stop = False
    for root, _, files in os.walk(directory_path):
        
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                # Step 2: Extract QA pairs from the file
                qa_pairs = extract_qa_pairs(file_path)
                
                # Step 3: Extend the list with extracted QA pairs
                all_qa_pairs.extend(qa_pairs)
                
                # Step 4: Stop if we have enough samples
                if len(all_qa_pairs) >= num_of_samples:
                    all_qa_pairs = all_qa_pairs[:num_of_samples]
                    stop = True
                    break
            
        if stop:
            break
    
    # Step 5: Convert to HuggingFace Datasets format
    hf_dataset = Dataset.from_list(all_qa_pairs)
    hf_dataset = hf_dataset.map(lambda x: preprocess_example(x, formatter), remove_columns=hf_dataset.column_names)
    hf_dataset = hf_dataset.map(apply_template, remove_columns=hf_dataset.column_names)
    
    return hf_dataset



# ======================= Training ========================== #

def printTrainer(trainer, tokenizer):
    train_dataloader = trainer.get_train_dataloader()

    for batch_data in train_dataloader:
        # Process only the first example in the batch
        input_ids = batch_data['input_ids'][0]
        attention_mask = batch_data['attention_mask'][0]
        label_ids = batch_data['labels'][0]

        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        decoded = tokenizer.decode(input_ids, skip_special_tokens=False)

        print("Tokens:")
        for i, token in enumerate(tokens):
            attn = attention_mask[i].item()
            label_id = label_ids[i].item()
            if label_id != -100:
                label_token = tokenizer.convert_ids_to_tokens([label_id])[0]
            else:
                label_token = 'IGN'

            print(f"{i:2d}: {token:12s} | Label_id: {label_id} | Attention: {attn} | Label: {label_token}")

        print("\nDecoded sentence:")
        print(decoded)
        break  # only process one batch

def get_max_seq_length_from_chat_dataset(dataset, tokenizer, instruction_template="<|im_start|>user\n", response_template="<|im_start|>assistant\n", percentile=97, verbose=True):
    """
    Computes tokenized sequence lengths for chat-style datasets and returns the recommended max_seq_length.
    """
    input_lengths = []

    for example in tqdm(dataset, desc="Tokenizing chat examples"):
        messages = example["messages"]
        full_text = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                full_text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                full_text += f"{instruction_template}{content}<|im_end|>\n"
            elif role == "assistant":
                full_text += f"{response_template}{content}<|im_end|>\n"
            else:
                raise ValueError(f"Unsupported role: {role}")

        # Tokenize without truncation to get length
        tokens = tokenizer(full_text, add_special_tokens=False, truncation=False)["input_ids"]
        input_lengths.append(len(tokens))

    if verbose:
        print(f"\n📊 Chat Token Stats:")
        print(f"  Min: {np.min(input_lengths)}")
        print(f"  Max: {np.max(input_lengths)}")
        print(f"  Mean: {np.mean(input_lengths):.2f}")
        print(f"  Median: {np.median(input_lengths)}")
        print(f"  {percentile}th Percentile: {np.percentile(input_lengths, percentile)}")

    return int(np.percentile(input_lengths, percentile))

def lora_config_and_args(ft_model_name, max_seq_length, rank_dimension=32, lora_alpha=8, lora_dropout=0.05):
    """
    Returns LoRA and training config, setting max_seq_length based on dataset stats.
    """
    peft_config = LoraConfig(
        r=rank_dimension,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    args = SFTConfig(
        max_seq_length=max_seq_length,
        output_dir=f"../Models/{ft_model_name}",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=2e-4,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True
    )

    return peft_config, args

def training(input_dir, num_of_samples, model_name, ft_model_name):
    #=====> Loading formatter
    formatter = RAGFormatter(directory='data/output/en-wikipedia-finance.txt')
    
    #=====> Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #=====> Load and process dataset
    dataset = process_dataset(directory_path=input_dir, num_of_samples=num_of_samples, formatter=formatter)

    #=====> Analyze token lengths using HF TRL’s chat-template format
    max_seq_length = get_max_seq_length_from_chat_dataset(dataset, tokenizer)

    #=====> Generate config with correct sequence length
    peft_config, args = lora_config_and_args(ft_model_name=ft_model_name, max_seq_length=max_seq_length)

    #=====> Data collator
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template="<|im_start|>user\n",
        response_template="<|im_start|>assistant\n",
        tokenizer=tokenizer,
        mlm=False
    )

    #=====> Trainer
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        data_collator=collator,
    )

    #=====> Print Trainer details
    print("Trainer Sample detail:")
    printTrainer(trainer, tokenizer)

    #=====> Train
    trainer.train()

    #=====> Save model
    trainer.save_model(f"ft_models/{ft_model_name}")
    trainer.push_to_hub(f"ibrahimbukhariLingua/{ft_model_name}")

    return f"ibrahimbukhariLingua/{ft_model_name}"



# ======================= Training Class ========================== #

class Finetune():
    
    def __init__(self, **kwargs):
        # Extract parameters from kwargs
        rag_format_dir = kwargs.get("rag_index_dir")
        input_dir = kwargs.get("input_dir", rag_format_dir)
        model_name = kwargs.get("model_name")
        device_map = kwargs.get("device_map", "auto")  # default to 'auto' if not provided
        num_of_samples = kwargs.get("num_of_samples", 1000)  # default to all samples if not specified
        self.ft_model_name = kwargs.get("ft_model_name")

        # Validate required parameters
        if not all([rag_format_dir, model_name, self.ft_model_name]):
            raise ValueError("Missing required parameters: 'rag_index_dir', 'model_name', or 'ft_model_name'.")

        # Initialize formatter and load model/tokenizer
        formatter = RAGFormatter(directory=rag_format_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
        self.model.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Prepare dataset and configuration
        self.dataset = process_dataset(
            directory_path=input_dir,
            num_of_samples=num_of_samples,
            formatter=formatter
        )
        self.max_seq_length = get_max_seq_length_from_chat_dataset(self.dataset, self.tokenizer)
        self.peft_config, self.args = lora_config_and_args(
            ft_model_name=self.ft_model_name,
            max_seq_length=self.max_seq_length
        )

    # -------- Helper Funtions -------- #
    
    def printTrainer(self, trainer, tokenizer):
        train_dataloader = trainer.get_train_dataloader()

        for batch_data in train_dataloader:
            # Process only the first example in the batch
            input_ids = batch_data['input_ids'][0]
            attention_mask = batch_data['attention_mask'][0]
            label_ids = batch_data['labels'][0]

            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            decoded = tokenizer.decode(input_ids, skip_special_tokens=False)

            print("Tokens:")
            for i, token in enumerate(tokens):
                attn = attention_mask[i].item()
                label_id = label_ids[i].item()
                if label_id != -100:
                    label_token = tokenizer.convert_ids_to_tokens([label_id])[0]
                else:
                    label_token = 'IGN'

                print(f"{i:2d}: {token:12s} | Label_id: {label_id} | Attention: {attn} | Label: {label_token}")

            print("\nDecoded sentence:")
            print(decoded)
            break
    
    def get_max_seq_length_from_chat_dataset(self, dataset, tokenizer, instruction_template="<|im_start|>user\n", response_template="<|im_start|>assistant\n", percentile=97, verbose=True):    
        """
        Computes tokenized sequence lengths for chat-style datasets and returns the recommended max_seq_length.
        """
        input_lengths = []

        for example in tqdm(dataset, desc="Tokenizing chat examples"):
            messages = example["messages"]
            full_text = ""
            for message in messages:
                role = message["role"]
                content = message["content"]
                if role == "system":
                    full_text += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    full_text += f"{instruction_template}{content}<|im_end|>\n"
                elif role == "assistant":
                    full_text += f"{response_template}{content}<|im_end|>\n"
                else:
                    raise ValueError(f"Unsupported role: {role}")

            # Tokenize without truncation to get length
            tokens = tokenizer(full_text, add_special_tokens=False, truncation=False)["input_ids"]
            input_lengths.append(len(tokens))

        if verbose:
            print(f"\n📊 Chat Token Stats:")
            print(f"  Min: {np.min(input_lengths)}")
            print(f"  Max: {np.max(input_lengths)}")
            print(f"  Mean: {np.mean(input_lengths):.2f}")
            print(f"  Median: {np.median(input_lengths)}")
            print(f"  {percentile}th Percentile: {np.percentile(input_lengths, percentile)}")

        return int(np.percentile(input_lengths, percentile))

    def lora_config_and_args(self, ft_model_name, max_seq_length, rank_dimension=32, lora_alpha=8, lora_dropout=0.05):
        """
        Returns LoRA and training config, setting max_seq_length based on dataset stats.
        """
        peft_config = LoraConfig(
            r=rank_dimension,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

        args = SFTConfig(
            max_seq_length=max_seq_length,
            output_dir=f"Models/{ft_model_name}",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            learning_rate=2e-4,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            logging_steps=10,
            save_strategy="epoch",
            bf16=True
        )

        return peft_config, args

    # -------- Training Function -------- #
    
    def run(self):
        #=====> Data collator
        collator = DataCollatorForCompletionOnlyLM(
            instruction_template="<|im_start|>user\n",
            response_template="<|im_start|>assistant\n",
            tokenizer=self.tokenizer,
            mlm=False
        )

        #=====> Trainer
        trainer = SFTTrainer(
            model=self.model,
            args=self.args,
            train_dataset=self.dataset,
            peft_config=self.peft_config,
            processing_class=self.tokenizer,
            data_collator=collator,
        )

        #=====> Print Trainer details
        print("Trainer Sample detail:")
        self.printTrainer(trainer, self.tokenizer)

        #=====> Train
        trainer.train()

        #=====> Save model
        trainer.save_model(f"ft_models/{self.ft_model_name}")
        trainer.push_to_hub(f"ibrahimbukhariLingua/{self.ft_model_name}")

        return f"ibrahimbukhariLingua/{self.ft_model_name}"




# ======================= Finetuning Class with Checkpoint Save ========================== #

class Finetune_w_checkpoint():
    
    def __init__(self, **kwargs):
        # Extract parameters from kwargs
        rag_format_dir = kwargs.get("rag_index_dir")
        input_dir = kwargs.get("input_dir", rag_format_dir)
        model_name = kwargs.get("model_name")
        device_map = kwargs.get("device_map", "auto")
        num_of_samples = kwargs.get("num_of_samples", 1000)
        self.ft_model_name = kwargs.get("ft_model_name")

        if not all([rag_format_dir, model_name, self.ft_model_name]):
            raise ValueError("Missing required parameters: 'rag_index_dir', 'model_name', or 'ft_model_name'.")

        formatter = RAGFormatter(directory=rag_format_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
        self.model.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.dataset = process_dataset(
            directory_path=input_dir,
            num_of_samples=num_of_samples,
            formatter=formatter
        )
        self.max_seq_length = self.get_max_seq_length_from_chat_dataset(self.dataset, self.tokenizer)

        total_steps = self.compute_total_steps(len(self.dataset))
        save_steps = total_steps // 2

        self.peft_config, self.args = self.lora_config_and_args(
            ft_model_name=self.ft_model_name,
            max_seq_length=self.max_seq_length,
            save_steps=save_steps
        )

    def compute_total_steps(self, num_samples, batch_size=1, gradient_accumulation_steps=8, epochs=1):
        steps_per_epoch = (num_samples // batch_size) // gradient_accumulation_steps
        return steps_per_epoch * epochs

    def printTrainer(self, trainer, tokenizer):
        train_dataloader = trainer.get_train_dataloader()
        for batch_data in train_dataloader:
            input_ids = batch_data['input_ids'][0]
            attention_mask = batch_data['attention_mask'][0]
            label_ids = batch_data['labels'][0]

            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            decoded = tokenizer.decode(input_ids, skip_special_tokens=False)

            print("Tokens:")
            for i, token in enumerate(tokens):
                attn = attention_mask[i].item()
                label_id = label_ids[i].item()
                label_token = tokenizer.convert_ids_to_tokens([label_id])[0] if label_id != -100 else 'IGN'
                print(f"{i:2d}: {token:12s} | Label_id: {label_id} | Attention: {attn} | Label: {label_token}")

            print("\nDecoded sentence:")
            print(decoded)
            break

    def get_max_seq_length_from_chat_dataset(self, dataset, tokenizer, instruction_template="<|im_start|>user\n", response_template="<|im_start|>assistant\n", percentile=97, verbose=True):    
        input_lengths = []
        for example in tqdm(dataset, desc="Tokenizing chat examples"):
            messages = example["messages"]
            full_text = ""
            for message in messages:
                role = message["role"]
                content = message["content"]
                if role == "system":
                    full_text += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    full_text += f"{instruction_template}{content}<|im_end|>\n"
                elif role == "assistant":
                    full_text += f"{response_template}{content}<|im_end|>\n"
                else:
                    raise ValueError(f"Unsupported role: {role}")

            tokens = tokenizer(full_text, add_special_tokens=False, truncation=False)["input_ids"]
            input_lengths.append(len(tokens))

        if verbose:
            print(f"\n📊 Chat Token Stats:")
            print(f"  Min: {np.min(input_lengths)}")
            print(f"  Max: {np.max(input_lengths)}")
            print(f"  Mean: {np.mean(input_lengths):.2f}")
            print(f"  Median: {np.median(input_lengths)}")
            print(f"  {percentile}th Percentile: {np.percentile(input_lengths, percentile)}")

        return int(np.percentile(input_lengths, percentile))

    def lora_config_and_args(self, ft_model_name, max_seq_length, rank_dimension=4, lora_alpha=8, lora_dropout=0.05, save_steps=100):
        peft_config = LoraConfig(
            r=rank_dimension,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

        args = SFTConfig(
            max_seq_length=max_seq_length,
            output_dir=f"Models/{ft_model_name}",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            learning_rate=2e-4,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            logging_steps=10,
            save_strategy="steps",              # Save based on step count
            save_steps=save_steps,              # Save at halfway point
            bf16=True
        )

        return peft_config, args

    def run(self):
        collator = DataCollatorForCompletionOnlyLM(
            instruction_template="<|im_start|>user\n",
            response_template="<|im_start|>assistant\n",
            tokenizer=self.tokenizer,
            mlm=False
        )

        trainer = SFTTrainer(
            model=self.model,
            args=self.args,
            train_dataset=self.dataset,
            peft_config=self.peft_config,
            processing_class=self.tokenizer,
            data_collator=collator,
        )

        print("Trainer Sample detail:")
        self.printTrainer(trainer, self.tokenizer)

        trainer.train()

        trainer.push_to_hub(f"ibrahimbukhariLingua/{self.ft_model_name}")

        return f"ibrahimbukhariLingua/{self.ft_model_name}"



# ======================= Main ========================== #

if __name__ == "__main__":
        
    from synthetic_data_kit.evaluation.util import RAGFormatter
    
    print("Loading formatter...")
    formatter = RAGFormatter(directory='data/output/en-wikipedia-finance.txt')
    print("Formatter loaded successfully.")
    
    print("Processing dataset...")
    dataset = process_dataset(directory_path="data/generated/en-wikipedia-finance.txt_pipeline_v2", num_of_samples=10, formatter=formatter)
    
    print(dataset[0][0]['content'])
    print("="* 80)
    print(dataset[0][1]['content'])
    print("="* 80)
    print(dataset[0][2]['content'])