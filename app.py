from flask import Flask, request, send_from_directory, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import whisper
from transformers import GPT4TurboTokenizer, GPT4TurboForConditionalGeneration


def generate_summary(text):
   model_name = 'gpt-4-turbo'
   tokenizer = GPT4TurboTokenizer.from_pretrained(model_name)
   model = GPT4TurboForConditionalGeneration.from_pretrained(model_name)

   prompt = "Summarize the following transcription: " + text + ". Provide a detailed explanation and cite sources where relevant."
   inputs = tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
   summary_ids = model.generate(inputs, max_length=150, min_length=80, length_penalty=5.0, num_beams=2)
   summary = tokenizer.decode(summary_ids[0])
   return summary

app = Flask(__name__)

@app.route('/')
def home():
   return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = secure_filename(file.filename)
    temp_folder = os.path.join('temp')
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    file.save(os.path.join(temp_folder, filename))
    return redirect(url_for('transcribe_file', filename=filename))

@app.route('/transcribe/<filename>', methods=['GET'])
def transcribe_file(filename):
    # filepath = os.path.join('temp', filename)
    # Load the model
    # model = whisper.load_model("base")
    # Transcribe the file
    # transcription = model.transcribe(filepath)
    # os.remove(filepath)
    
    # Read the transcription from the file
    with open('transcription.txt', 'r', encoding='utf-8') as file:
        transcription_text = file.read()
    
    # Generate a summary
    summary = generate_summary(transcription_text)
    return render_template('result.html', transcription=transcription_text, summary=summary)


if __name__ == '__main__':
    app.run(debug=True)