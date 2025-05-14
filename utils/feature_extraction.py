from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(df, text_column):
    """Extract TF-IDF features from the specified text column."""
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df[text_column])
    return X, tfidf
