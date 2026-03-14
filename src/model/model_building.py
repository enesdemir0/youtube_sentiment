import numpy as np
import pandas as pd
import os
import yaml
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# --- TEACHER'S LOGGING (EXACT) ---
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def load_params(params_path: str) -> dict:
    with open(params_path, 'r') as file:
        return yaml.safe_load(file)

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.fillna('', inplace=True)
    # BINGO: Modern models need the labels to be integers
    df['label'] = df['label'].astype(int)
    return df

def get_root_directory() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

def main():
    try:
        root_dir = get_root_directory()
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        
        # Modern Params
        model_name = params['model_building']['model_name']
        epochs = params['model_building']['epochs']
        batch_size = params['model_building']['batch_size']
        lr = params['model_building']['learning_rate']

        # 1. Load Data
        train_df = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))
        
        # BINGO: To save time on your laptop, we take a small sample first!
        # You can remove .sample(1000) when you want to train on EVERYTHING
        train_df = train_df.sample(1000) 
        
        # 2. Tokenization (The Modern way to process text)
        logger.debug(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        # Convert Pandas to Hugging Face Dataset
        train_dataset = Dataset.from_pandas(train_df)
        tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

        # 3. Load Model
        logger.debug(f"Loading model: {model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        # 4. Training Arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(root_dir, "models/results"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            logging_dir=os.path.join(root_dir, "logs"),
            report_to="none" # BINGO: We will use MLflow later!
        )

        # 5. The Trainer (The "Boss" of modern AI)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        logger.info("Starting model training...")
        trainer.train()

        # 6. Save Model and Tokenizer
        model_save_path = os.path.join(root_dir, 'models/sentiment_model')
        os.makedirs(model_save_path, exist_ok=True)
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        logger.info(f"BINGO! Model saved to {model_save_path}")

    except Exception as e:
        logger.error('Failed in Model Building: %s', e)
        raise

if __name__ == '__main__':
    main()