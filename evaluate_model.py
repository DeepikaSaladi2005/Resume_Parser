'''import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Importing custom modules from the utils folder
from utils.load_data import load_data
from utils.preprocess_text import preprocess_dataframe
from utils.feature_extraction import extract_features

def train_and_evaluate_model(X, y):
    """Train a logistic regression model and evaluate it using classification report."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, y_test, y_pred, report

def plot_confusion_matrix(y_test, y_pred, classes):
    """Plot the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig("plots/confusion_matrix.png")
    plt.show()

def plot_classification_report(report):
    """Plot bar graphs for precision, recall, and F1-score for each class."""
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df = metrics_df.drop(['accuracy', 'macro avg', 'weighted avg'], axis=0)
    metrics = ["precision", "recall", "f1-score"]
    
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        sns.barplot(x=metrics_df.index, y=metrics_df[metric], palette="viridis")
        plt.title(metric.capitalize())
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/classification_report_metrics.png")
    plt.show()

if __name__ == "__main__":
    # Load and preprocess data
    csv_file = 'data/resumes_dataset.csv'
    df = load_data(csv_file)
    df = preprocess_dataframe(df, 'Resume')
    
    # Extract features and labels
    X, tfidf = extract_features(df, 'Resume')
    y = df['Category']
    
    # Train model and get evaluation results
    model, y_test, y_pred, report = train_and_evaluate_model(X, y)
    
    # Print classification report to console
    print(pd.DataFrame(report).transpose())
    
    # Plot confusion matrix
    unique_classes = y.unique()
    plot_confusion_matrix(y_test, y_pred, unique_classes)
    
    # Plot classification report metrics
    plot_classification_report(report)'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Importing custom modules from the utils folder
from utils.load_data import load_data
from utils.preprocess_text import preprocess_dataframe
from utils.feature_extraction import extract_features

def train_and_evaluate_model(X, y, test_size):
    """Train a logistic regression model and evaluate it using classification report."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report['accuracy']
    return model, y_test, y_pred, report, accuracy

def plot_results(test_sizes, accuracies):
    """Plot accuracies for different test sizes."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=test_sizes, y=accuracies, marker='o')
    plt.title('Model Accuracy vs Test Size')
    plt.xlabel('Test Size')
    plt.ylabel('Accuracy')
    plt.xticks(test_sizes)
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig("plots/accuracy_vs_test_size.png")
    plt.show()

if __name__ == "__main__":
    # Load and preprocess data
    csv_file = 'data/resumes_dataset.csv'
    df = load_data(csv_file)
    df = preprocess_dataframe(df, 'Resume')
    
    # Extract features and labels
    X, tfidf = extract_features(df, 'Resume')
    y = df['Category']
    
    # Define different test sizes to evaluate
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9]
    accuracies = []

    # Train model and evaluate for each test size
    for size in test_sizes:
        model, y_test, y_pred, report, accuracy = train_and_evaluate_model(X, y, size)
        accuracies.append(accuracy)
        print(f'Test Size: {size}, Accuracy: {accuracy:.4f}')

    # Plot results
    plot_results(test_sizes, accuracies)

    # You can add additional plotting functions for confusion matrix and classification report metrics if needed


