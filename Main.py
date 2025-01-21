import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import requests
import json
from datetime import datetime

# Ensure your OpenRouter API key is set
openrouter_api_key = os.getenv('openrouter_api_key')

# Load the Hugging Face model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"An error occurred while extracting text from PDF: {e}")
        return None

def summarize_text(text):
    try:
        prompt = f"""Summarize the following file:

        CONTEXT:
        {text}
        """
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": "google/gemini-2.0-flash-thinking-exp:free",
                "messages": [
                    {"role": "user", 
                    "content": prompt}
                ]
            })
        )

        if response.status_code == 200:
            llm_response = response.json().get("choices", [])[0].get('message', {}).get("content", None)
            if llm_response:
                return llm_response
            else:
                print("Error: No content returned from the model.")
                return None
        else:
            print(f"Error: Failed to get a valid response from the model. Status code: {response.status_code}")
            print("Response content:", response.content)
            return None
        
    except Exception as e:
        print("Error:", str(e))
        return None

def save_summary(summary):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_directory = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(project_directory, f"Summary_{timestamp}.txt")
        
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(summary)
        
        print(f"Summary saved to {output_file}")
      
    except Exception as e:
        print("Error:", str(e))

# Example usage
file_path = input("Enter the path to the PDF file: ")
transcript = extract_text_from_pdf(file_path)
if transcript:
    summary = summarize_text(transcript)
    if summary:
        save_summary(summary)
