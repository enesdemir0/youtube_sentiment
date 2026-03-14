import numpy as np
import pandas as pd
import os
import yaml
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def load_params(params_path: str) -> dict:
    with open(params_path, 'r') as file:
        return yaml.safe_load(file)

def get_root_directory() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

def save_confusion_matrix(cm, root_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Neg', 'Neu', 'Pos'], 
                yticklabels=['Neg', 'Neu', 'Pos'])
    plt.title('Confusion Matrix - YouTube Sentiment')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    path = os.path.join(root_dir, 'confusion_matrix.png')
    plt.savefig(path)
    plt.close()
    logger.info(f"Confusion matrix saved to {path}")

def main():
    try:
        root_dir = get_root_directory()
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        
        # BINGO: Pointing to your LOCAL model folder
        model_path = os.path.join(root_dir, 'models/sentiment_model')
        test_data_path = os.path.join(root_dir, 'data/interim/test_processed.csv')

        # 1. Load Model and Tokenizer from your local disk
        logger.debug(f"Loading local model from {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 2. Load Test Data
        test_df = pd.read_csv(test_data_path).dropna()
        # Take a small sample (e.g., 100) just to verify it works fast
        test_df = test_df.sample(100) 
        
        # 3. Predict
        logger.info("Running predictions on test data...")
        inputs = tokenizer(list(test_df['text']), padding=True, truncation=True, return_tensors="pt")
        
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).numpy()
        
        y_test = test_df['label'].values

        # 4. Calculate Metrics
        acc = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        print("\n--- MODEL PERFORMANCE REPORT ---")
        print(report)

        # 5. Save Results Locally 
        metrics = {
            "accuracy": float(acc),
            "model_name": params['model_building']['model_name']
        }
        
        with open(os.path.join(root_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        save_confusion_matrix(cm, root_dir)
        
        logger.info("BINGO! Local evaluation complete.")

    except Exception as e:
        logger.error(f"Local Evaluation Failed: {e}")
        raise

if __name__ == '__main__':
    main()