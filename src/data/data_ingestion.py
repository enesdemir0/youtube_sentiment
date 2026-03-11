import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from datasets import load_dataset  # BINGO! Added this
import yaml
import logging

# --- TEACHER'S LOGGING CONFIGURATION (EXACT) ---
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading params: %s', e)
        raise

def load_data() -> pd.DataFrame:
    """Load from Hugging Face and convert Dict to DataFrame."""
    try:
        logger.debug('Loading dataset from Hugging Face: tweet_eval/sentiment')
        # Your data is a DatasetDict. We take the 'train' part to split it ourselves
        dataset = load_dataset("tweet_eval", "sentiment")
        df = pd.DataFrame(dataset['train']) 
        
        logger.debug('Data loaded from Hugging Face successfully.')
        return df
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data using YOUR column name 'text'."""
    try:
        # Removing missing values
        df.dropna(inplace=True)
        # Removing duplicates
        df.drop_duplicates(inplace=True)
        
        #Your column is called 'text', not 'clean_comment'
        df = df[df['text'].str.strip() != '']
        
        logger.debug('Data preprocessing completed: Missing values, duplicates, and empty strings removed.')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        # 1. Load parameters (Teacher's Path)
        # This path looks for params.yaml two levels up from src/data/
        params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml')
        params = load_params(params_path)
        test_size = params['data_ingestion']['test_size']
        
        # 2. Load data from Hugging Face
        df = load_data()
        
        # 3. Preprocess the data
        final_df = preprocess_data(df)
        
        # 4. Split the data (Enes manually controls the split via params.yaml)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        
        # 5. Save the split datasets
        data_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')
        save_data(train_data, test_data, data_path=data_save_path)
        
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()