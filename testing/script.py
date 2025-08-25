from flask import Flask, request, jsonify
import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizer
from safetensors.torch import load_file
import re
import requests
import PyPDF2
import io
import os
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
from waitress import serve
import logging
from anthropic import Anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Global variables
global model, tokenizer, device

# Model classes
class DualTruthfulnessClassifier(torch.nn.Module):
    def __init__(self, hidden_size, num_labels=2):
        super(DualTruthfulnessClassifier, self).__init__()
        self.token_classifier = torch.nn.Linear(hidden_size, num_labels)
        self.sentence_classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states):
        token_logits = self.token_classifier(hidden_states)
        sentence_logits = self.sentence_classifier(hidden_states[:, 0, :])
        return token_logits, sentence_logits

class BERTForDualTruthfulness(torch.nn.Module):
    def __init__(self, bert_model, hidden_size, num_labels=2):
        super(BERTForDualTruthfulness, self).__init__()
        self.bert = bert_model
        self.dual_classifier = DualTruthfulnessClassifier(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.last_hidden_state
        token_logits, sentence_logits = self.dual_classifier(hidden_states)
        return token_logits, sentence_logits, hidden_states

def initialize_model_and_tokenizer():
    global model, tokenizer, device
    
    saved_model_dir = "."
    model_path = os.path.join(saved_model_dir, "model.safetensors")

    config = BertConfig.from_pretrained("bert-base-uncased")
    bert_model = BertModel(config)
    model = BERTForDualTruthfulness(bert_model, hidden_size=config.hidden_size)

    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained(saved_model_dir)
    logger.info("Model and tokenizer loaded successfully!")

def download_pdf(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    
    if 'application/pdf' not in response.headers.get('content-type', '').lower():
        if 'arxiv.org' in url and '/pdf/' not in url:
            url = url.replace('/abs/', '/pdf/') + '.pdf'
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
    
    return io.BytesIO(response.content)

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    # Remove references section
    ref_patterns = [r'References\s*\n', r'REFERENCES\s*\n']
    min_ref_index = len(text)
    for pattern in ref_patterns:
        match = re.search(pattern, text)
        if match and match.start() < min_ref_index:
            min_ref_index = match.start()
    
    return text[:min_ref_index]

def simple_sent_tokenize(text):
    return re.split(r'(?<=[.!?])\s+', text)

def get_sentence_embedding(sentence):
    global model, tokenizer, device
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        _, _, hidden_states = model(**inputs)
    return hidden_states[:, 0, :].squeeze().cpu().numpy()

def get_truthfulness_score(sentence):
    global model, tokenizer, device
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        _, sentence_logits, _ = model(**inputs)
    sentence_score = torch.softmax(sentence_logits, dim=-1)[:, 1].item()
    return sentence_score

def get_uncertainty_score(sentence):
    return 1 - get_truthfulness_score(sentence)

def get_context_relevance(sentences, index, window_size=2):
    start = max(0, index - window_size)
    end = min(len(sentences), index + window_size + 1)
    context = sentences[start:end]
    
    target_embedding = get_sentence_embedding(sentences[index])
    context_embeddings = [get_sentence_embedding(sent) for sent in context if sent != sentences[index]]
    
    similarities = [1 - cosine(target_embedding, ctx_emb) for ctx_emb in context_embeddings]
    avg_similarity = np.mean(similarities) if similarities else 0
    
    return (avg_similarity + 1) / 2

def get_novelty_summary(sentence, paper_text, score):
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    prompt = f"""Based on the following novel sentence from an academic paper, generate a concise abstract.

Paper context: {paper_text[:500]}...
Novel sentence: "{sentence}"
Novelty Score: {score:.2f}

Write an abstract (150-200 words) that:
1. Introduces the research question suggested by the novel sentence
2. Outlines a potential methodology
3. Speculates on possible results
4. Emphasizes the novelty and significance

Abstract:"""

    response = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
        
    try:
        # Download and extract PDF
        pdf_file = download_pdf(url)
        paper_text = extract_text_from_pdf(pdf_file)
        
        # Analyze sentences
        sentences = simple_sent_tokenize(paper_text)
        novelty_scores = []
        
        for i, sentence in enumerate(sentences):
            uncertainty = get_uncertainty_score(sentence)
            relevance = get_context_relevance(sentences, i)
            score = uncertainty * relevance
            if score > 0.4:  # Minimum threshold
                novelty_scores.append((sentence, score, i))
        
        # Get top 5 novel sentences
        novelty_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = novelty_scores[:5]
        
        # Generate results
        results = []
        for sentence, score, idx in top_sentences:
            abstract = get_novelty_summary(sentence, paper_text, score)
            results.append({
                "sentence": sentence,
                "score": float(score),
                "abstract": abstract,
                "position": idx
            })
        
        return jsonify({
            "status": "success",
            "results": results,
            "total_sentences": len(sentences)
        })

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    initialize_model_and_tokenizer()
    # Run locally only
    serve(app, host='127.0.0.1', port=8888, threads=4)
