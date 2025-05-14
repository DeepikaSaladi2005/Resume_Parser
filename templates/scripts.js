// scripts.js

document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('files');
    const skillsInput = document.getElementById('skills');
    const messageTextarea = document.getElementById('message');

    // Simple validation check
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length === 0) {
            alert('Please upload at least one file.');
        }
    });

    document.querySelector('form').addEventListener('submit', (event) => {
        if (!skillsInput.value.trim()) {
            alert('Please enter at least one skill.');
            event.preventDefault();
        }
        if (!messageTextarea.value.trim()) {
            alert('Email content cannot be empty.');
            event.preventDefault();
        }
    });
});
