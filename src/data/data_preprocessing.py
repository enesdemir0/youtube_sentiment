import numpy as np
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# LOGGING CONFIGURATION  ---
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_comment(comment):
    """BINGO: Deep cleaning logic for YouTube comments."""
    try:
        if not isinstance(comment, str):
            return ""
            
        # 1. Convert to lowercase
        comment = comment.lower()

        # 2. Remove URLs (Very important for social media!)
        comment = re.sub(r'http\S+|www\S+|https\S+', '', comment, flags=re.MULTILINE)

        # 3. Remove @user tags
        comment = re.sub(r'@\w+', '', comment)

        # 4. Remove non-alphanumeric characters (keep basic punctuation for sentiment)
        comment = re.sub(r'[^a-z0-9\s!?.,]', '', comment)

        # 5. Remove stopwords but keep words like 'not' and 'but' 
        # (These are CRITICAL for sentiment!)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'no', 'never', 'don', 'didn', 'doesn'}
        words = comment.split()
        comment = ' '.join([word for word in words if word not in stop_words])

        # 6. Lemmatize (Turning 'running' into 'run')
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment.strip()
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment

def normalize_text(df):
    """BINGO: Apply cleaning to YOUR column named 'text'."""
    try:
        # Check if column exists
        if 'text' not in df.columns:
            logger.error("Column 'text' not found in DataFrame!")
            raise KeyError("Column 'text' not found")

        df['text'] = df['text'].apply(preprocess_comment)
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save to 'interim' folder as the teacher did."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        os.makedirs(interim_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)
        
        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

def main():
    try:
        logger.debug("Starting data preprocessing stage...")
        
        # Load the raw data created in the Ingestion stage
        # BINGO: We use the paths relative to the project root
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Raw data loaded successfully')

        # Clean the text
        train_processed = normalize_text(train_data)
        test_processed = normalize_text(test_data)

        # Save to interim folder
        save_data(train_processed, test_processed, data_path='./data')
        logger.info("BINGO! Preprocessing stage finished successfully.")
        
    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()