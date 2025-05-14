import pandas as pd

# Load resumes dataset
csv_file = '../data/resumes_dataset.csv'  # Adjusted path based on your project structure
df = pd.read_csv(csv_file)

# Define skills of interest
desired_skills = ['Python', 'Machine Learning']

# Function to check if a resume contains any of the desired skills
def has_desired_skills(resume_text):
    for skill in desired_skills:
        if skill.lower() in resume_text.lower():
            return True
    return False

# Filter resumes based on desired skills
relevant_resumes = []
for index, row in df.iterrows():
    if has_desired_skills(row['Resume']):
        relevant_resumes.append(row['Resume'])

# Print relevant resumes
for resume in relevant_resumes:
    print(resume)
    print('-' * 50)
