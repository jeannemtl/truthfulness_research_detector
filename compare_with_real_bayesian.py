"""
Compare Introspective vs Bayesian using the REAL Bayesian Flask API model
This uses your properly trained BERTForBayesianDualTruthfulness
"""

import torch
from transformers import BertTokenizer, BertModel, BertConfig
import sys
import os
import json
import numpy as np
from anthropic import Anthropic
from safetensors.torch import load_file
from scipy.spatial.distance import cosine

sys.path.append('/workspace/truthfulness_research_detector')
from model.architectures.introspective_uncertainty import BERTForIntrospectiveUncertainty

# =============================================================================
# BAYESIAN MODEL (from Flask API)
# =============================================================================

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

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_introspective_model():
    """Load introspective multi-task model"""
    model_path = "/workspace/truthfulness_research_detector/uncertainty_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = BertTokenizer.from_pretrained(model_path)
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = BERTForIntrospectiveUncertainty(bert_model, hidden_size=768)
    
    state_dict = load_file(os.path.join(model_path, 'model.safetensors'))
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def load_bayesian_model():
    """Load REAL Bayesian model with dropout"""
    model_path = "/workspace/truthfulness_research_detector/uncertainty_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = BertConfig.from_pretrained("bert-base-uncased")
    bert_model = BertModel(config)
    
    model = BERTForBayesianDualTruthfulness(
        bert_model, 
        hidden_size=config.hidden_size,
        dropout_rate=0.1
    )
    
    state_dict = load_file(os.path.join(model_path, 'model.safetensors'))
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    return model, tokenizer, device

# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def get_sentence_embedding_introspective(sentence, model, tokenizer, device):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        _, _, _, _, _, hidden_states = outputs
    return hidden_states[:, 0, :].squeeze().cpu().numpy()

def get_sentence_embedding_bayesian(sentence, model, tokenizer, device):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(device)
    model.eval()
    with torch.no_grad():
        _, _, hidden_states = model(**inputs)
    return hidden_states[:, 0, :].squeeze().cpu().numpy()

def get_context_relevance(sentences, index, embeddings_cache, model, tokenizer, device, method='introspective'):
    start = max(0, index - 10)
    end = min(len(sentences), index + 11)
    
    if index not in embeddings_cache:
        if method == 'introspective':
            embeddings_cache[index] = get_sentence_embedding_introspective(sentences[index], model, tokenizer, device)
        else:
            embeddings_cache[index] = get_sentence_embedding_bayesian(sentences[index], model, tokenizer, device)
    
    target_embedding = embeddings_cache[index]
    similarities = []
    
    for i in range(start, end):
        if i == index:
            continue
        if i not in embeddings_cache:
            if method == 'introspective':
                embeddings_cache[i] = get_sentence_embedding_introspective(sentences[i], model, tokenizer, device)
            else:
                embeddings_cache[i] = get_sentence_embedding_bayesian(sentences[i], model, tokenizer, device)
        
        similarity = 1 - cosine(target_embedding, embeddings_cache[i])
        similarities.append(similarity)
    
    avg_similarity = np.mean(similarities) if similarities else 0
    return float((avg_similarity + 1) / 2)

# =============================================================================
# INTROSPECTIVE ANALYSIS
# =============================================================================

def get_introspective_scores(sentence, model, tokenizer, device):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        (token_logits, sentence_logits, uncertainty_logits, 
         conflict_logits, epistemic_logits, hidden_states) = outputs
    
    truth_probs = torch.softmax(sentence_logits, dim=-1)[0]
    uncertainty_probs = torch.softmax(uncertainty_logits, dim=-1)[0]
    conflict_probs = torch.softmax(conflict_logits, dim=-1)[0]
    
    return {
        'truth_prob': float(truth_probs[1].item()),
        'hedging_score': float(uncertainty_probs[1].item() + uncertainty_probs[2].item()),
        'conflict_score': float(conflict_probs[1].item()),
    }

def analyze_introspective(paper_text, model, tokenizer, device):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', paper_text)
    
    interesting = []
    embeddings_cache = {}
    
    research_keywords = [
        'could', 'future', 'investigate', 'explore', 'unclear', 'remains',
        'question', 'whether', 'possible', 'potential', 'further', 'warrant',
        'should', 'might', 'may', 'speculate', 'limitation'
    ]
    
    for i, sentence in enumerate(sentences):
        if len(sentence.split()) < 10:
            continue
        
        try:
            scores = get_introspective_scores(sentence, model, tokenizer, device)
            has_keyword = any(kw in sentence.lower() for kw in research_keywords)
            
            if (scores['hedging_score'] > 0.4 or scores['conflict_score'] > 0.4) and has_keyword:
                relevance = get_context_relevance(
                    sentences, i, embeddings_cache, model, tokenizer, device, 'introspective'
                )
                
                combined = (scores['hedging_score'] + scores['conflict_score']) * relevance
                
                interesting.append({
                    'sentence': sentence,
                    'combined_score': float(combined),
                    'relevance': float(relevance),
                    **scores
                })
        except:
            continue
    
    interesting.sort(key=lambda x: x['combined_score'], reverse=True)
    return interesting[:10]

# =============================================================================
# BAYESIAN ANALYSIS (using REAL model)
# =============================================================================

def get_bayesian_uncertainty(sentence, model, tokenizer, device, num_samples=50):
    """Monte Carlo Dropout with REAL Bayesian model"""
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(device)
    
    predictions = []
    model.train()  # Enable dropout
    
    with torch.no_grad():
        for _ in range(num_samples):
            _, sentence_logits, _ = model(**inputs)
            pred = torch.softmax(sentence_logits, dim=-1)[:, 1].item()
            predictions.append(pred)
    
    model.eval()
    
    mean = np.mean(predictions)
    uncertainty = np.std(predictions)
    
    return {
        'truthfulness': float(mean),
        'uncertainty': float(uncertainty),
        'confidence_interval': (
            float(max(0, mean - 1.96 * uncertainty)),
            float(min(1, mean + 1.96 * uncertainty))
        )
    }

def analyze_bayesian(paper_text, model, tokenizer, device):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', paper_text)
    
    interesting = []
    embeddings_cache = {}
    
    research_keywords = [
        'could', 'future', 'investigate', 'explore', 'unclear', 'remains',
        'question', 'whether', 'possible', 'potential', 'further', 'warrant',
        'should', 'might', 'may', 'speculate', 'limitation'
    ]
    
    print(f"  Processing {len(sentences)} sentences...")
    
    for i, sentence in enumerate(sentences):
        if len(sentence.split()) < 10:
            continue
        
        try:
            has_keyword = any(kw in sentence.lower() for kw in research_keywords)
            if not has_keyword:
                continue
                
            truth_result = get_bayesian_uncertainty(sentence, model, tokenizer, device)
            
            # ANY uncertainty is interesting
            if truth_result['uncertainty'] > 0.01:  # Very low threshold
                relevance = get_context_relevance(
                    sentences, i, embeddings_cache, model, tokenizer, device, 'bayesian'
                )
                
                # Weighted combination
                truth_weight = 1 / (truth_result['uncertainty'] + 0.01)
                rel_weight = 10.0
                
                novelty_score = (
                    (truth_result['truthfulness'] * truth_weight + relevance * rel_weight) /
                    (truth_weight + rel_weight)
                )
                
                interesting.append({
                    'sentence': sentence,
                    'novelty_score': float(novelty_score),
                    'uncertainty': float(truth_result['uncertainty']),
                    'confidence_interval': truth_result['confidence_interval'],
                    'relevance': float(relevance)
                })
                
                if i % 100 == 0 and len(interesting) > 0:
                    print(f"    Found {len(interesting)} candidates at sentence {i}")
        except Exception as e:
            continue
    
    interesting.sort(key=lambda x: x['uncertainty'], reverse=True)  # Sort by MOST uncertain
    print(f"  Total candidates: {len(interesting)}")
    return interesting[:10]

# =============================================================================
# IDEA GENERATION
# =============================================================================

def generate_idea_claude45(statement_data, method_name):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key)
    
    if method_name == "introspective":
        uncertainty_info = f"""Introspective Uncertainty:
- Hedging: {statement_data['hedging_score']:.3f}
- Conflict: {statement_data['conflict_score']:.3f}
- Combined: {statement_data['combined_score']:.3f}"""
    else:
        uncertainty_info = f"""Bayesian Uncertainty (Monte Carlo Dropout):
- Novelty: {statement_data['novelty_score']:.3f}
- Epistemic Uncertainty: {statement_data['uncertainty']:.4f}
- CI: [{statement_data['confidence_interval'][0]:.3f}, {statement_data['confidence_interval'][1]:.3f}]"""
    
    prompt = f"""Generate ONE research idea from this uncertain statement.

Statement: "{statement_data['sentence']}"

{uncertainty_info}

Return ONLY valid JSON:
{{
  "Name": "snake_case",
  "Title": "Research Title",
  "Experiment": "Plan (3 sentences)",
  "Interestingness": 8,
  "Feasibility": 8,
  "Novelty": 8
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    
    try:
        content = response.content[0].text
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0]
        elif '```' in content:
            content = content.split('```')[1].split('```')[0]
        return json.loads(content.strip())
    except:
        return None

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("INTROSPECTIVE vs BAYESIAN (REAL MODELS)")
    print("="*80)
    
    # Load paper
    print("\n[1/6] Loading paper...")
    with open("language_modeling_compression.txt", 'r') as f:
        paper_text = f.read()
    print(f"✓ Loaded {len(paper_text)} characters")
    
    # INTROSPECTIVE
    print("\n[2/6] Loading introspective model...")
    intro_model, intro_tokenizer, intro_device = load_introspective_model()
    print("✓ Loaded")
    
    print("\n[3/6] Analyzing with introspective...")
    intro_statements = analyze_introspective(paper_text, intro_model, intro_tokenizer, intro_device)
    print(f"✓ Found {len(intro_statements)} statements")
    
    print("\n  Generating introspective ideas...")
    intro_ideas = []
    for i, stmt in enumerate(intro_statements[:8], 1):
        print(f"    Idea {i}/8...")
        idea = generate_idea_claude45(stmt, "introspective")
        if idea:
            idea['method'] = 'introspective'
            idea['source'] = stmt['sentence'][:200]
            idea['scores'] = {
                'hedging': stmt['hedging_score'],
                'conflict': stmt['conflict_score'],
                'combined': stmt['combined_score']
            }
            intro_ideas.append(idea)
    
    # BAYESIAN
    print("\n[4/6] Loading Bayesian model...")
    bayes_model, bayes_tokenizer, bayes_device = load_bayesian_model()
    print("✓ Loaded")
    
    print("\n[5/6] Analyzing with Bayesian...")
    bayes_statements = analyze_bayesian(paper_text, bayes_model, bayes_tokenizer, bayes_device)
    print(f"✓ Found {len(bayes_statements)} statements")
    
    print("\n  Generating Bayesian ideas...")
    bayes_ideas = []
    for i, stmt in enumerate(bayes_statements[:8], 1):
        print(f"    Idea {i}/8...")
        idea = generate_idea_claude45(stmt, "bayesian")
        if idea:
            idea['method'] = 'bayesian'
            idea['source'] = stmt['sentence'][:200]
            idea['scores'] = {
                'novelty': stmt['novelty_score'],
                'uncertainty': stmt['uncertainty']
            }
            bayes_ideas.append(idea)
    
    # SAVE
    print("\n[6/6] Saving...")
    results = {
        'introspective': {
            'ideas': intro_ideas,
            'count': len(intro_ideas)
        },
        'bayesian': {
            'ideas': bayes_ideas,
            'count': len(bayes_ideas)
        }
    }
    
    with open('final_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print(f"Introspective: {len(intro_ideas)} ideas")
    print(f"Bayesian: {len(bayes_ideas)} ideas")
    print("\nSaved: final_comparison.json")

if __name__ == "__main__":
    main()
