import re

def preprocess_dataframe(df, text_column):
    """Clean and preprocess the text data in the specified column."""
    def clean_text(text):
        # Basic text cleaning operations
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    df[text_column] = df[text_column].apply(clean_text)
    return df
