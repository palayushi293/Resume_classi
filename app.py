from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  
from PyPDF2 import PdfReader
import re
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load models
MODEL_PATH ="C:\\Users\\palay\\OneDrive\\Desktop\\react\\comment_analyzer"  
try:
    rf_classifier_categorization = pickle.load(open(os.path.join(MODEL_PATH, 'rf_classifier_categorization.pkl'), 'rb'))
    tfidf_vectorizer_categorization = pickle.load(open(os.path.join(MODEL_PATH, 'tfidf_vectorizer_categorization.pkl'), 'rb'))
    print("Models loaded successfully.")
except Exception as e:
    rf_classifier_categorization = None
    tfidf_vectorizer_categorization = None
    print(f"Error loading models: {e}")

# Clean resume text
def clean_resume(txt):
    txt = re.sub(r'http\S+', ' ', txt)
    txt = re.sub(r'RT|cc', ' ', txt)
    txt = re.sub(r'#\S+', ' ', txt)
    txt = re.sub(r'@\S+', ' ', txt)
    txt = re.sub(r'[^\w\s]', ' ', txt)  
    txt = re.sub(r'\s+', ' ', txt).strip()

def predict_category(resume_text):
    if tfidf_vectorizer_categorization is None or rf_classifier_categorization is None:
        return "Error: Model not loaded."
    
    resume_text = clean_resume(resume_text)
    try:
        resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
        predicted_category = rf_classifier_categorization.predict(resume_tfidf)[0]
        return predicted_category
    except Exception as e:
        return f"Prediction error: {e}"


def pdf_to_text(file):
    try:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text if text.strip() else "No text extracted from this PDF."
    except Exception as e:
        return f"Error reading PDF: {e}"

@app.route("/")
def home():
    return render_template("resume.html", title="Resume Categorization")

@app.route('/predict', methods=['POST'])
def predict():
    if 'resume' not in request.files or request.files['resume'].filename == '':
        return jsonify({"error": "No file uploaded or invalid file."})
    
    file = request.files['resume']
    extracted_text = pdf_to_text(file)
    
    if extracted_text.startswith("Error"):
        return jsonify({"error": extracted_text})
    else:
        predicted_category = predict_category(extracted_text)
        return jsonify({"predicted_category": predicted_category})

if __name__ == '__main__':
    app.run(debug=True)
