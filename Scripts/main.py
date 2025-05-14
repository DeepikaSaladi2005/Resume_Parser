# main.py
import pandas as pd
from scripts.load_data import load_data
from scripts.preprocess_text import preprocess_dataframe
from scripts.explore_data import plot_category_distribution
from scripts.feature_extraction import extract_features
from scripts.model import train_and_evaluate_model

csv_file = 'data/resumes_dataset.csv'

# Load data
df = load_data(csv_file)

# Preprocess text
df = preprocess_dataframe(df, 'Resume')

# Explore data
plot_category_distribution(df, 'Category')

# Extract features
X, tfidf = extract_features(df, 'Resume')

# Train and evaluate model
y = df['Category']
model, report = train_and_evaluate_model(X, y)
print(report)
