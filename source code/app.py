from flask import Flask, request, render_template, redirect, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
import os
import PyPDF2
import docx
import re
import smtplib
from email.mime.text import MIMEText
from calculate_accuracy import calculate_accuracy  # Import calculate_accuracy from the new file

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

def extract_emails(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(email_pattern, text)

def send_email(sender_email, sender_password, receiver_email, subject, message):
    try:
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = receiver_email

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.sendmail(sender_email, receiver_email, msg.as_string())

        print(f"Email sent successfully to {receiver_email}!")
    except Exception as e:
        print(f"Failed to send email to {receiver_email}: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    skills = request.form['skills'].split(',')

    session['skills'] = skills
    skill_resumes = {skill.strip().lower(): [] for skill in skills}

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            if file_path.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif file_path.endswith('.docx'):
                text = extract_text_from_docx(file_path)

            if all(re.search(r'\b' + re.escape(skill.strip()) + r'\b', text, re.IGNORECASE) for skill in skills):
                skill_resumes[', '.join(skills).strip().lower()].append(filename)

                emails = extract_emails(text)
                session['emails'] = emails

    session['skill_resumes'] = skill_resumes

    # Call calculate_accuracy and print the result to the terminal
    dataset_path = 'data/resumes_dataset.csv'  # Adjust path if necessary
    accuracy = calculate_accuracy(skill_resumes, skills, dataset_path)
    print(f"Calculated Accuracy: {accuracy:.2f}%")  # Display accuracy in the terminal

    return redirect(url_for('email_page'))

@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/email', methods=['GET', 'POST'])
def email_page():
    if request.method == 'POST':
        message = request.form['message']
        session['email_content'] = message

        sender_email = "zorogoku072@gmail.com"
        sender_password = "nqzb rdrz cpmf opjz"  # Replace with your actual app-specific password

        emails = session.get('emails', [])
        for email in emails:
            send_email(sender_email, sender_password, email, "Hello from Python!", message)

        return redirect(url_for('results'))

    return render_template('email.html')

@app.route('/results')
def results():
    skill_resumes = session.get('skill_resumes', {})
    return render_template('results.html', skill_resumes=skill_resumes)

if __name__ == "__main__":
    app.run(debug=True)


