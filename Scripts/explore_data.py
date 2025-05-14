# scripts/explore_data.py
import matplotlib.pyplot as plt

def plot_category_distribution(df, category_column):
    plt.figure(figsize=(10, 6))
    df[category_column].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Distribution of Categories')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    import pandas as pd
    from load_data import load_data

    csv_file = '../data/resumes_dataset.csv'
    df = load_data(csv_file)
    plot_category_distribution(df, 'Category')
