import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import docx
import PyPDF2

# --- Load Models ---
def load_models():
    ai_tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")
    ai_model = AutoModelForSequenceClassification.from_pretrained("openai-community/roberta-base-openai-detector")
    fake_news_pipeline = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
    return ai_tokenizer, ai_model, fake_news_pipeline

ai_tokenizer, ai_model, fake_news_pipeline = load_models()

# --- Functions ---
def extract_text(file):
    if file is None:
        return ""
    
    filename = file.name.lower()
    text = ""
    
    if filename.endswith(".pdf"):
        pdf = PyPDF2.PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    elif filename.endswith(".docx"):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        # Assume plain text
        text = file.read().decode("utf-8")
    
    return text.strip()

def analyze_text(text, uploaded_file=None):
    # If file uploaded, extract text
    if uploaded_file is not None:
        text = extract_text(uploaded_file)
    
    if not text.strip():
        return "⚠️ Please enter text or upload a file!", None, None

    # AI-generated Detection
    inputs = ai_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = ai_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    ai_result = {
        "Human Probability": round(float(probs[0][0]), 2),
        "AI Generated Probability": round(float(probs[0][1]), 2)
    }

    # Fake News Detection
    fake_result = fake_news_pipeline(text)

    return text, ai_result, fake_result

# --- Custom CSS for professional look ---
custom_css = """
body {
    background: linear-gradient(135deg, #FFEB3B, #9C27B0);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #1a1a1a;
}
h1, h2, h3, h4 {
    text-align: center;
    color: #1a1a1a;
}
.gr-button {
    background-color: #4B0082 !important;
    color: white !important;
    border-radius: 8px !important;
}
"""

# --- Gradio Interface ---
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🦀 TextGuard AI Fake Content Detector", elem_id="title")
    gr.Markdown("Detect whether text is AI-generated or contains fake news using AI models.", elem_id="subtitle")

    with gr.Row():
        text_input = gr.Textbox(label="Paste your text here", lines=10, placeholder="Type or paste your text...")
        file_input = gr.File(label="Or upload a PDF / DOCX / TXT file", file_types=[".pdf", ".docx", ".txt"])
        analyze_btn = gr.Button("🔎 Analyze")

    with gr.Row():
        text_output = gr.Textbox(label="Original Text")
        ai_output = gr.JSON(label="AI Generated Detection")
        fake_output = gr.JSON(label="Fake News Detection")

    analyze_btn.click(fn=analyze_text, inputs=[text_input, file_input], outputs=[text_output, ai_output, fake_output])

demo.launch()
