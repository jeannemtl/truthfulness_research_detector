from flask import Flask, request, jsonify
import numpy as np
import torch
import torch.distributions as dist
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

global model, tokenizer, device

class BayesianDualTruthfulnessClassifier(torch.nn.Module):
    def __init__(self, hidden_size, num_labels=2, dropout_rate=0.1):
        super(BayesianDualTruthfulnessClassifier, self).__init__()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.token_classifier = torch.nn.Linear(hidden_size, num_labels)
        self.sentence_classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states):
        # Apply dropout for uncertainty estimation
        dropped_states = self.dropout(hidden_states)
        token_logits = self.token_classifier(dropped_states)
        sentence_logits = self.sentence_classifier(dropped_states[:, 0, :])
        return token_logits, sentence_logits

class BERTForBayesianDualTruthfulness(torch.nn.Module):
    def __init__(self, bert_model, hidden_size, num_labels=2, dropout_rate=0.1):
        super(BERTForBayesianDualTruthfulness, self).__init__()
        self.bert = bert_model
        self.dual_classifier = BayesianDualTruthfulnessClassifier(
            hidden_size, num_labels, dropout_rate
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        hidden_states = outputs.last_hidden_state
        token_logits, sentence_logits = self.dual_classifier(hidden_states)
        return token_logits, sentence_logits, hidden_states

def initialize_model_and_tokenizer():
    global model, tokenizer, device
    
    saved_model_dir = "."
    model_path = os.path.join(saved_model_dir, "model.safetensors")

    config = BertConfig.from_pretrained("bert-base-uncased")
    bert_model = BertModel(config)
    
    # Use Bayesian version with dropout for uncertainty
    model = BERTForBayesianDualTruthfulness(
        bert_model, 
        hidden_size=config.hidden_size,
        dropout_rate=0.1
    )

    state_dict = load_file(model_path)
    model.load_state_dict(state_dict, strict=False)  # Allow missing dropout layers
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained(saved_model_dir)
    logger.info("Bayesian model and tokenizer loaded successfully!")

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
    model.eval()
    with torch.no_grad():
        _, _, hidden_states = model(**inputs)
    return hidden_states[:, 0, :].squeeze().cpu().numpy()

def get_bayesian_uncertainty_score(sentence, num_samples=50):
    """Calculate epistemic uncertainty using Monte Carlo Dropout"""
    global model, tokenizer, device
    
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(device)
    
    predictions = []
    model.train()  # Enable dropout for uncertainty estimation
    
    with torch.no_grad():
        for _ in range(num_samples):
            _, sentence_logits, _ = model(**inputs)
            pred = torch.softmax(sentence_logits, dim=-1)[:, 1].item()
            predictions.append(pred)
    
    model.eval()  # Back to eval mode
    
    # Calculate statistics
    mean_pred = np.mean(predictions)
    epistemic_uncertainty = np.std(predictions)
    
    return mean_pred, epistemic_uncertainty, predictions

def get_bayesian_truthfulness_score(sentence):
    """Get truthfulness with Bayesian confidence intervals"""
    mean_truth, uncertainty, samples = get_bayesian_uncertainty_score(sentence)
    
    # Calculate 95% confidence interval
    confidence_interval = 1.96 * uncertainty
    lower_bound = max(0, mean_truth - confidence_interval)
    upper_bound = min(1, mean_truth + confidence_interval)
    
    return {
        'truthfulness': mean_truth,
        'uncertainty': uncertainty,
        'confidence_interval': (lower_bound, upper_bound),
        'confidence_width': upper_bound - lower_bound,
        'samples': len(samples)
    }

def get_bayesian_context_relevance(sentences, index, window_size=2):
    """Bayesian approach to context relevance with uncertainty"""
    start = max(0, index - window_size)
    end = min(len(sentences), index + window_size + 1)
    context = sentences[start:end]
    
    target_embedding = get_sentence_embedding(sentences[index])
    similarities = []
    
    for sent in context:
        if sent != sentences[index] and len(sent.strip()) > 10:  # Skip very short sentences
            try:
                ctx_emb = get_sentence_embedding(sent)
                similarity = 1 - cosine(target_embedding, ctx_emb)
                # Normalize similarity to [0, 1] range
                similarity = max(0, min(1, (similarity + 1) / 2))
                similarities.append(similarity)
            except Exception as e:
                logger.warning(f"Error calculating similarity: {e}")
                continue
    
    if not similarities:
        return {'relevance': 0.5, 'uncertainty': 0.5}
    
    # Use Beta distribution for bounded probabilities
    # Transform similarities to be more suitable for Beta distribution
    alpha = sum(similarities) + 1  # Add prior
    beta = len(similarities) - sum(similarities) + 1
    
    # Ensure positive parameters
    alpha = max(alpha, 0.1)
    beta = max(beta, 0.1)
    
    try:
        beta_dist = dist.Beta(alpha, beta)
        mean_relevance = beta_dist.mean.item()
        uncertainty = beta_dist.variance.item() ** 0.5
    except Exception:
        # Fallback to simple statistics
        mean_relevance = np.mean(similarities)
        uncertainty = np.std(similarities) if len(similarities) > 1 else 0.1
    
    return {
        'relevance': mean_relevance,
        'uncertainty': min(uncertainty, 0.5),  # Cap uncertainty
        'context_size': len(similarities)
    }

def get_bayesian_novelty_score(sentence, sentences, index):
    """Combine multiple Bayesian estimates for novelty"""
    
    # Skip very short sentences
    if len(sentence.strip()) < 20:
        return None
    
    try:
        # Get Bayesian truthfulness
        truth_result = get_bayesian_truthfulness_score(sentence)
        
        # Get Bayesian context relevance
        relevance_result = get_bayesian_context_relevance(sentences, index)
        
        # Calculate uncertainty-weighted combination
        truth_weight = 1 / (truth_result['uncertainty'] + 0.01)
        relevance_weight = 1 / (relevance_result['uncertainty'] + 0.01)
        
        total_weight = truth_weight + relevance_weight
        
        # Weighted combination
        novelty_score = (
            (truth_result['truthfulness'] * truth_weight + 
             relevance_result['relevance'] * relevance_weight) / total_weight
        )
        
        # Combined uncertainty using error propagation
        combined_uncertainty = np.sqrt(
            (truth_result['uncertainty'] * truth_weight / total_weight) ** 2 +
            (relevance_result['uncertainty'] * relevance_weight / total_weight) ** 2
        )
        
        # Calculate confidence interval
        ci_width = 1.96 * combined_uncertainty
        lower_bound = max(0, novelty_score - ci_width)
        upper_bound = min(1, novelty_score + ci_width)
        
        return {
            'novelty_score': novelty_score,
            'uncertainty': combined_uncertainty,
            'confidence_interval': (lower_bound, upper_bound),
            'confidence_width': upper_bound - lower_bound,
            'components': {
                'truthfulness': truth_result,
                'relevance': relevance_result
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating Bayesian novelty score: {e}")
        return None

def get_novelty_summary(sentence, paper_text, score, uncertainty):
    """Generate summary with uncertainty information"""
    try:
        anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        confidence_level = "high" if uncertainty < 0.1 else "medium" if uncertainty < 0.2 else "low"
        
        prompt = f"""Based on the following novel sentence from an academic paper, generate a concise abstract.

Paper context: {paper_text[:500]}...
Novel sentence: "{sentence}"
Novelty Score: {score:.2f} (Confidence: {confidence_level}, Uncertainty: {uncertainty:.2f})

Write an abstract (150-200 words) that:
1. Introduces the research question suggested by the novel sentence
2. Outlines a potential methodology
3. Speculates on possible results
4. Emphasizes the novelty and significance
5. Notes the confidence level of this assessment

Abstract:"""

        response = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return f"High novelty sentence detected with {confidence_level} confidence. Further analysis recommended."

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    url = data.get('url')
    confidence_threshold = data.get('confidence_threshold', 0.3)  # User can set threshold
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
        
    try:
        # Download and extract PDF
        logger.info(f"Processing PDF from URL: {url}")
        pdf_file = download_pdf(url)
        paper_text = extract_text_from_pdf(pdf_file)
        
        # Analyze sentences
        sentences = simple_sent_tokenize(paper_text)
        logger.info(f"Analyzing {len(sentences)} sentences")
        
        novelty_results = []
        
        for i, sentence in enumerate(sentences):
            if i % 100 == 0:  # Progress logging
                logger.info(f"Processed {i}/{len(sentences)} sentences")
                
            bayesian_result = get_bayesian_novelty_score(sentence, sentences, i)
            
            if bayesian_result is None:
                continue
                
            # Filter based on novelty score and confidence
            if (bayesian_result['novelty_score'] > 0.4 and 
                bayesian_result['uncertainty'] < confidence_threshold):
                
                novelty_results.append((sentence, bayesian_result, i))
        
        # Sort by uncertainty-adjusted novelty score
        novelty_results.sort(
            key=lambda x: x[1]['novelty_score'] - x[1]['uncertainty'], 
            reverse=True
        )
        
        top_sentences = novelty_results[:5]
        results = []
        
        for sentence, bayesian_result, idx in top_sentences:
            abstract = get_novelty_summary(
                sentence, 
                paper_text, 
                bayesian_result['novelty_score'],
                bayesian_result['uncertainty']
            )
            
            results.append({
                "sentence": sentence,
                "novelty_score": float(bayesian_result['novelty_score']),
                "uncertainty": float(bayesian_result['uncertainty']),
                "confidence_interval": [float(x) for x in bayesian_result['confidence_interval']],
                "confidence_width": float(bayesian_result['confidence_width']),
                "adjusted_score": float(bayesian_result['novelty_score'] - bayesian_result['uncertainty']),
                "abstract": abstract,
                "position": idx,
                "components": {
                    "truthfulness": {
                        "score": float(bayesian_result['components']['truthfulness']['truthfulness']),
                        "uncertainty": float(bayesian_result['components']['truthfulness']['uncertainty']),
                        "confidence_width": float(bayesian_result['components']['truthfulness']['confidence_width'])
                    },
                    "relevance": {
                        "score": float(bayesian_result['components']['relevance']['relevance']),
                        "uncertainty": float(bayesian_result['components']['relevance']['uncertainty']),
                        "context_size": bayesian_result['components']['relevance']['context_size']
                    }
                }
            })
        
        return jsonify({
            "status": "success",
            "results": results,
            "total_sentences": len(sentences),
            "analyzed_sentences": len(novelty_results),
            "methodology": "bayesian_monte_carlo_dropout",
            "parameters": {
                "confidence_threshold": confidence_threshold,
                "monte_carlo_samples": 50,
                "context_window": 2
            }
        })

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "methodology": "bayesian_uncertainty_quantification",
        "features": [
            "monte_carlo_dropout",
            "confidence_intervals", 
            "epistemic_uncertainty",
            "bayesian_combination"
        ]
    })

if __name__ == "__main__":
    initialize_model_and_tokenizer()
    logger.info("Starting Bayesian uncertainty-aware analysis server...")
    serve(app, host='127.0.0.1', port=8888, threads=4)
