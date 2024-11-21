from flask import Flask, request, render_template
import PyPDF2
import os
from transformers import pipeline

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Summarization pipeline
summarizer = pipeline("summarization", model="t5-small")

# Helper function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Helper function to summarize text
def summarize_text(text, max_length=300, min_length=50):
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Splitter Function
def split_text(text, chunk_size=500):
    sentences = text.split('.')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        token_length = len(sentence.split())
        if current_length + token_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += token_length

    if current_chunk:  # Add the last chunk if any
        chunks.append(" ".join(current_chunk))

    return chunks


# Route for uploading PDF and generating summary
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.pdf'):
            # Save the uploaded PDF
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Extract text from the PDF
            text = extract_text_from_pdf(filepath)

            # Summarize the extracted text
            # Limiting to the first 1000 characters for performance (can be adjusted)
            summaries = [summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
             for chunk in split_text(text)]
            
            final_summary = " ".join(summaries)

            return render_template('index.html', text=text, summary=final_summary)

    # Default case: Render upload form
    return render_template('index.html', text=None, summary=None)

if __name__ == '__main__':
    app.run(debug=True)
