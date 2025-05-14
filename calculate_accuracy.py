import pandas as pd

def calculate_accuracy(skill_resumes, skills, dataset_path='resume_dataset.csv'):
    # Load the resume dataset
    data = pd.read_csv(dataset_path)

    # Filter resumes based on the provided skills and calculate the total matches
    total_resumes = len(data)
    matching_resumes = 0
    for skill in skills:
        matching_resumes += len(skill_resumes.get(skill.lower(), []))

    # Calculate accuracy as a percentage of matching resumes
    accuracy = (matching_resumes / total_resumes) * 100 if total_resumes > 0 else 0
    return accuracy
