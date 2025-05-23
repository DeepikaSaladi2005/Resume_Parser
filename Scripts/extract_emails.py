# scripts/extract_emails.py
import re

def extract_emails(text):
    # Regex pattern to find emails
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    return emails
