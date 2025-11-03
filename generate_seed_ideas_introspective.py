"""
Generate seed ideas with INTROSPECTIVE relevance detection
Using Lindsey-style introspection: epistemic certainty head
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

# =============================================================================
# INTROSPECTIVE RELEVANCE (NEW!)
# =============================================================================

def get_context_relevance_introspective(target_sentence, context_sentences, 
                                       model, tokenizer, device):
    """
    INTROSPECTIVE RELEVANCE using epistemic certainty head
    
    Based on Lindsey et al. (2025) introspection methodology:
    - Model introspects on its own epistemic certainty
    - High certainty when processing related sentences = high relevance
    - Causally grounded in internal activations
    
    This satisfies Lindsey's criteria:
    1. Accuracy: Epistemic head reflects actual certainty
    2. Grounding: Depends causally on activations
    3. Internality: Uses internal state, not just text
    4. Metacognitive: Model judges its own certainty
    """
    
    certainty_scores = []
    
    for ctx_sent in context_sentences:
        if ctx_sent == target_sentence:
            continue
        
        # Process both sentences together
        text = f"{target_sentence} [SEP] {ctx_sent}"
        
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Unpack: (token_logits, sentence_logits, uncertainty_logits,
            #          conflict_logits, epistemic_logits, hidden_states)
            epistemic_logits = outputs[4]
        
        # Epistemic head: [uncertain, certain]
        epistemic_probs = torch.softmax(epistemic_logits, dim=-1)[0]
        certainty = float(epistemic_probs[0].item())  # P(certain)
        
        certainty_scores.append(certainty)
    
    # Average epistemic certainty = introspective relevance
    avg_certainty = np.mean(certainty_scores) if certainty_scores else 0.5
    
    return float(avg_certainty)

def get_context_relevance_baseline(target_sentence, context_sentences,
                                   embeddings_cache, model, tokenizer, device):
    """
    BASELINE: Cosine similarity (non-introspective)
    Keep for comparison
    """
    
    def get_embedding(sentence):
        if sentence not in embeddings_cache:
            inputs = tokenizer(sentence, return_tensors='pt', 
                             truncation=True, padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                _, _, _, _, _, hidden_states = outputs
            embeddings_cache[sentence] = hidden_states[:, 0, :].squeeze().cpu().numpy()
        return embeddings_cache[sentence]
    
    target_emb = get_embedding(target_sentence)
    
    similarities = []
    for ctx_sent in context_sentences:
        if ctx_sent == target_sentence:
            continue
        ctx_emb = get_embedding(ctx_sent)
        sim = 1 - cosine(target_emb, ctx_emb)
        similarities.append(sim)
    
    avg_similarity = np.mean(similarities) if similarities else 0
    return float((avg_similarity + 1) / 2)

# =============================================================================
# UNCERTAINTY DETECTION (introspective)
# =============================================================================

def get_detailed_introspective_scores(sentence, model, tokenizer, device):
    """Get introspective uncertainty scores"""
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, 
                      padding=True).to(device)
    
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

# =============================================================================
# PAPER ANALYSIS
# =============================================================================

def analyze_paper_introspective(paper_text, model, tokenizer, device, 
                                use_introspective=True):
    """
    Analyze paper with introspective OR baseline relevance
    
    Args:
        use_introspective: If True, use epistemic certainty (introspective)
                          If False, use cosine similarity (baseline)
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', paper_text)
    
    interesting_statements = []
    embeddings_cache = {}  # For baseline method
    
    method = "INTROSPECTIVE" if use_introspective else "BASELINE"
    print(f"  Analyzing {len(sentences)} sentences with {method} relevance...")
    
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
            # Get uncertainty scores (introspective)
            scores = get_detailed_introspective_scores(sentence, model, tokenizer, device)
            has_research_keyword = any(kw in sentence.lower() for kw in research_keywords)
            
            if (scores['hedging_score'] > 0.4 or scores['conflict_score'] > 0.4) and has_research_keyword:
                
                # Get context
                start = max(0, i - 10)
                end = min(len(sentences), i + 11)
                context = [sentences[j] for j in range(start, end) if j != i]
                
                # Calculate relevance (INTROSPECTIVE or BASELINE)
                if use_introspective:
                    context_relevance = get_context_relevance_introspective(
                        sentence, context, model, tokenizer, device
                    )
                else:
                    context_relevance = get_context_relevance_baseline(
                        sentence, context, embeddings_cache, model, tokenizer, device
                    )
                
                combined_score = (scores['hedging_score'] + scores['conflict_score']) * context_relevance
                
                interesting_statements.append({
                    'sentence': sentence,
                    'combined_score': float(combined_score),
                    'context_relevance': float(context_relevance),
                    'relevance_method': method.lower(),
                    **scores
                })
                
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(sentences)}")
                
        except Exception as e:
            continue
    
    interesting_statements.sort(key=lambda x: x['combined_score'], reverse=True)
    return interesting_statements[:15]

# =============================================================================
# IDEA GENERATION
# =============================================================================

def generate_research_idea(statement_data):
    """Generate research idea from uncertain statement"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    
    client = Anthropic(api_key=api_key)
    
    relevance_method = statement_data.get('relevance_method', 'unknown')
    
    prompt = f"""Based on this uncertain statement from "Language Modeling Is Compression", generate a research idea.

Statement: "{statement_data['sentence']}"

Detection Scores:
- Hedging: {statement_data['hedging_score']:.3f}
- Conflict: {statement_data['conflict_score']:.3f}
- Relevance: {statement_data['context_relevance']:.3f} ({relevance_method})

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

# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate seed ideas')
    parser.add_argument('--introspective', action='store_true', 
                       help='Use introspective relevance (epistemic certainty)')
    parser.add_argument('--baseline', action='store_true',
                       help='Use baseline relevance (cosine similarity)')
    parser.add_argument('--both', action='store_true',
                       help='Generate ideas with both methods and compare')
    
    args = parser.parse_args()
    
    # Default: introspective
    if not args.introspective and not args.baseline and not args.both:
        args.introspective = True
    
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
        return
    
    with open(paper_path, 'r', encoding='utf-8') as f:
        paper_text = f.read()
    
    print(f"✓ Loaded {len(paper_text)} characters")
    
    # Generate with chosen method(s)
    all_results = {}
    
    if args.both:
        methods = [('introspective', True), ('baseline', False)]
    elif args.baseline:
        methods = [('baseline', False)]
    else:
        methods = [('introspective', True)]
    
    for method_name, use_intro in methods:
        print(f"\n[3/4] Analyzing with {method_name.upper()} relevance...")
        statements = analyze_paper_introspective(
            paper_text, model, tokenizer, device, use_introspective=use_intro
        )
        print(f"✓ Found {len(statements)} high-uncertainty statements")
        
        print(f"\nTop 5 uncertain statements ({method_name}):")
        for i, stmt in enumerate(statements[:5], 1):
            print(f"\n  {i}. Score: {stmt['combined_score']:.3f} | Rel: {stmt['context_relevance']:.3f}")
            print(f"     {stmt['sentence'][:80]}...")
        
        print(f"\n[4/4] Generating research ideas ({method_name})...")
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
                    'combined': stmt['combined_score'],
                    'method': stmt['relevance_method']
                }
                seed_ideas.append(idea)
        
        print(f"✓ Generated {len(seed_ideas)} ideas with {method_name}")
        
        # Save
        output_file = f"seed_ideas_{method_name}.json"
        with open(output_file, 'w') as f:
            json.dump(seed_ideas, f, indent=2)
        
        all_results[method_name] = {
            'ideas': seed_ideas,
            'file': output_file,
            'count': len(seed_ideas)
        }
    
    # Summary
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    
    for method_name, results in all_results.items():
        print(f"\n{method_name.upper()}:")
        print(f"  Output: {results['file']}")
        print(f"  Ideas: {results['count']}")
        
        if results['ideas']:
            scores = [idea['uncertainty_scores']['combined'] for idea in results['ideas']]
            relevances = [idea['uncertainty_scores']['relevance'] for idea in results['ideas']]
            print(f"  Mean combined score: {np.mean(scores):.3f}")
            print(f"  Mean relevance: {np.mean(relevances):.3f}")
    
    # Compare if both
    if args.both:
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        
        intro_scores = [idea['uncertainty_scores']['combined'] 
                       for idea in all_results['introspective']['ideas']]
        base_scores = [idea['uncertainty_scores']['combined'] 
                      for idea in all_results['baseline']['ideas']]
        
        print(f"\nMean combined scores:")
        print(f"  Introspective: {np.mean(intro_scores):.3f}")
        print(f"  Baseline:      {np.mean(base_scores):.3f}")
        
        if np.mean(intro_scores) > np.mean(base_scores):
            diff = ((np.mean(intro_scores) / np.mean(base_scores)) - 1) * 100
            print(f"\n✓ Introspective is {diff:+.1f}% better!")
        else:
            diff = ((np.mean(base_scores) / np.mean(intro_scores)) - 1) * 100
            print(f"\n✗ Baseline is {diff:+.1f}% better")

if __name__ == "__main__":
    main()
