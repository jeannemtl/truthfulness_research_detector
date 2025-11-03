"""
Generate seed ideas from "Language Modeling Is Compression" paper
Using introspective uncertainty detection
"""

import torch
from transformers import BertTokenizer, BertModel
import sys
import os
import json
import numpy as np
from anthropic import Anthropic
from safetensors.torch import load_file
from scipy.spatial.distance import cosine

sys.path.append('/workspace/truthfulness_research_detector')
from model.architectures.introspective_uncertainty import BERTForIntrospectiveUncertainty

def load_model():
    """Load the trained introspective uncertainty model"""
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

def get_sentence_embedding(sentence, model, tokenizer, device):
    """Get sentence embedding from [CLS] token"""
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        _, _, _, _, _, hidden_states = outputs
    return hidden_states[:, 0, :].squeeze().cpu().numpy()

def get_context_relevance(sentences, index, embeddings_cache, model, tokenizer, device, window_size=10):
    """Calculate context relevance"""
    start = max(0, index - window_size)
    end = min(len(sentences), index + window_size + 1)
    
    if index not in embeddings_cache:
        embeddings_cache[index] = get_sentence_embedding(sentences[index], model, tokenizer, device)
    target_embedding = embeddings_cache[index]
    
    similarities = []
    for i in range(start, end):
        if i == index:
            continue
        if i not in embeddings_cache:
            embeddings_cache[i] = get_sentence_embedding(sentences[i], model, tokenizer, device)
        
        similarity = 1 - cosine(target_embedding, embeddings_cache[i])
        similarities.append(similarity)
    
    avg_similarity = np.mean(similarities) if similarities else 0
    return float((avg_similarity + 1) / 2)

def get_detailed_introspective_scores(sentence, model, tokenizer, device):
    """Get introspective uncertainty scores"""
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        (token_logits, sentence_logits, uncertainty_logits, 
         conflict_logits, epistemic_logits, hidden_states) = outputs
    
    truth_probs = torch.softmax(sentence_logits, dim=-1)[0]
    uncertainty_probs = torch.softmax(uncertainty_logits, dim=-1)[0]
    conflict_probs = torch.softmax(conflict_logits, dim=-1)[0]
    
    scores = {
        'truth_prob': float(truth_probs[1].item()),
        'hedging_score': float(uncertainty_probs[1].item() + uncertainty_probs[2].item()),
        'conflict_score': float(conflict_probs[1].item()),
    }
    
    return scores

def analyze_paper(paper_text, model, tokenizer, device):
    """Analyze Language Modeling Is Compression paper"""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', paper_text)
    
    interesting_statements = []
    embeddings_cache = {}
    
    print(f"  Analyzing {len(sentences)} sentences...")
    
    research_keywords = [
        'could', 'future', 'investigate', 'explore', 'unclear', 'remains',
        'question', 'whether', 'possible', 'potential', 'further', 'warrant',
        'should', 'might', 'may', 'speculate', 'limitation', 'open question',
        'challenge', 'difficult', 'unknown', 'hope', 'believe', 'suggest',
        'hypothesize', 'conjecture', 'unclear', 'remains to be seen'
    ]
    
    for i, sentence in enumerate(sentences):
        if len(sentence.split()) < 10:
            continue
        
        try:
            scores = get_detailed_introspective_scores(sentence, model, tokenizer, device)
            has_research_keyword = any(kw in sentence.lower() for kw in research_keywords)
            
            if (scores['hedging_score'] > 0.4 or scores['conflict_score'] > 0.4) and has_research_keyword:
                context_relevance = get_context_relevance(
                    sentences, i, embeddings_cache, model, tokenizer, device
                )
                
                combined_score = (scores['hedging_score'] + scores['conflict_score']) * context_relevance
                
                interesting_statements.append({
                    'sentence': sentence,
                    'combined_score': float(combined_score),
                    'context_relevance': float(context_relevance),
                    **scores
                })
                
            if i % 200 == 0:
                print(f"  Progress: {i}/{len(sentences)}")
        except Exception as e:
            continue
    
    interesting_statements.sort(key=lambda x: x['combined_score'], reverse=True)
    return interesting_statements[:15]

def generate_research_idea(statement_data):
    """Generate research idea from uncertain statement"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    
    client = Anthropic(api_key=api_key)
    
    prompt = f"""Based on this uncertain statement from "Language Modeling Is Compression", generate a research idea.

Statement: "{statement_data['sentence']}"

Uncertainty Analysis:
- Hedging: {statement_data['hedging_score']:.3f}
- Conflict: {statement_data['conflict_score']:.3f}
- Relevance: {statement_data['context_relevance']:.3f}

Generate research idea as JSON:
{{
  "Name": "descriptive_snake_case",
  "Title": "Clear Research Title",
  "Experiment": "Concrete experimental plan (2-3 sentences)",
  "Interestingness": 7-10,
  "Feasibility": 6-9,
  "Novelty": 7-10
}}

Focus on compression, language modeling, or information theory connections.
Return ONLY valid JSON."""

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
    except Exception as e:
        print(f"  ⚠ Parse error: {e}")
        return None

def main():
    print("="*80)
    print("LANGUAGE MODELING IS COMPRESSION - PAPER ANALYSIS")
    print("="*80)
    
    print("\n[1/4] Loading model...")
    model, tokenizer, device = load_model()
    print("✓ Model loaded")
    
    print("\n[2/4] Loading paper...")
    paper_path = "language_modeling_compression.txt"
    
    if not os.path.exists(paper_path):
        print(f"⚠ Paper not found at {paper_path}")
        print("Run: wget https://arxiv.org/pdf/2310.06824.pdf && extract text")
        return
    
    with open(paper_path, 'r', encoding='utf-8') as f:
        paper_text = f.read()
    
    print(f"✓ Loaded {len(paper_text)} characters")
    
    print("\n[3/4] Analyzing for uncertainty + relevance...")
    statements = analyze_paper(paper_text, model, tokenizer, device)
    print(f"✓ Found {len(statements)} high-uncertainty statements")
    
    print("\nTop 5 uncertain statements:")
    for i, stmt in enumerate(statements[:5], 1):
        print(f"\n  {i}. (score: {stmt['combined_score']:.3f})")
        print(f"     {stmt['sentence'][:100]}...")
    
    print("\n[4/4] Generating research ideas...")
    seed_ideas = []
    
    for i, stmt in enumerate(statements[:8], 1):
        print(f"  Idea {i}/8...")
        idea = generate_research_idea(stmt)
        if idea:
            idea['source_statement'] = stmt['sentence']
            idea['uncertainty_scores'] = {
                'hedging': stmt['hedging_score'],
                'conflict': stmt['conflict_score'],
                'relevance': stmt['context_relevance'],
                'combined': stmt['combined_score']
            }
            seed_ideas.append(idea)
    
    print(f"✓ Generated {len(seed_ideas)} ideas")
    
    # Save
    with open("seed_ideas.json", 'w') as f:
        json.dump(seed_ideas, f, indent=2)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nOutput: seed_ideas.json")
    print(f"Ideas generated: {len(seed_ideas)}")
    
    if seed_ideas:
        print("\nExample idea:")
        print(json.dumps(seed_ideas[0], indent=2))

if __name__ == "__main__":
    main()
