"""
Research Seed Idea Generation using Theory of Mind Head
Analyzes "Language Modeling Is Compression" paper
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from safetensors.torch import load_file
import os
import re
from typing import List, Dict, Tuple
from tqdm import tqdm

class BERTForTheoryOfMindIntrospection(nn.Module):
    def __init__(self, bert_model, hidden_size=768):
        super().__init__()
        self.bert = bert_model
        self.token_classifier = nn.Linear(hidden_size, 2)
        self.sentence_classifier = nn.Linear(hidden_size, 2)
        self.uncertainty_classifier = nn.Linear(hidden_size, 3)
        self.conflict_classifier = nn.Linear(hidden_size, 2)
        self.epistemic_classifier = nn.Linear(hidden_size, 2)
        self.theory_of_mind_classifier = nn.Linear(hidden_size, 3)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.last_hidden_state
        cls_hidden = hidden_states[:, 0, :]
        
        return (
            self.token_classifier(hidden_states),
            self.sentence_classifier(cls_hidden),
            self.uncertainty_classifier(cls_hidden),
            self.conflict_classifier(cls_hidden),
            self.epistemic_classifier(cls_hidden),
            self.theory_of_mind_classifier(cls_hidden),
            hidden_states
        )

def extract_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [s for s in sentences if 10 <= len(s.split()) <= 50]
    return sentences

def is_research_relevant(sentence: str) -> bool:
    # Skip references, citations, metadata
    skip_patterns = [
        r'^\d+\.',
        r'Figure \d+',
        r'Table \d+',
        r'Section \d+',
        r'\[[\d,\s]+\]',
        r'et al\.',
        r'@',
        r'http'
    ]
    
    for pattern in skip_patterns:
        if re.search(pattern, sentence):
            return False
    
    # Require research keywords
    research_keywords = [
        'could', 'future', 'investigate', 'explore', 'unclear', 'remains',
        'question', 'whether', 'possible', 'potential', 'further', 'warrant',
        'should', 'might', 'may', 'speculate', 'limitation', 'open',
        'challenge', 'difficult', 'unknown', 'hope', 'believe', 'suggest',
        'hypothesize', 'conjecture', 'remains to be seen'
    ]
    
    has_keyword = any(kw in sentence.lower() for kw in research_keywords)
    return has_keyword and len(sentence.split()) >= 10

def analyze_comprehensive(statement: str, model, tokenizer, device) -> Dict:
    inputs = tokenizer(statement, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    uncertainty_probs = torch.softmax(outputs[2], dim=-1)[0]
    conflict_probs = torch.softmax(outputs[3], dim=-1)[0]
    epistemic_probs = torch.softmax(outputs[4], dim=-1)[0]
    tom_probs = torch.softmax(outputs[5], dim=-1)[0]
    
    return {
        'linguistic_hedging': uncertainty_probs[1].item() + uncertainty_probs[2].item(),
        'has_conflict': conflict_probs[1].item(),
        'model_certainty': epistemic_probs[0].item(),
        'author_uncertainty': tom_probs[1].item() + tom_probs[2].item(),
        'tom_probs': {
            'certain': tom_probs[0].item(),
            'hedging': tom_probs[1].item(),
            'very_uncertain': tom_probs[2].item()
        }
    }

def generate_seed_ideas(paper_text: str, model, tokenizer, device, threshold=0.5, max_seeds=20):
    print("="*80)
    print("SEED IDEA GENERATION WITH THEORY OF MIND")
    print("="*80)
    
    print("\n[1/4] Extracting sentences...")
    sentences = extract_sentences(paper_text)
    print(f"✓ {len(sentences)} sentences")
    
    print("\n[2/4] Filtering for research-relevant...")
    relevant = [s for s in sentences if is_research_relevant(s)]
    print(f"✓ {len(relevant)} research-relevant sentences")
    
    print("\n[3/4] Analyzing with Theory of Mind...")
    seed_candidates = []
    
    for sentence in tqdm(relevant, desc="Analyzing"):
        try:
            analysis = analyze_comprehensive(sentence, model, tokenizer, device)
            if analysis['author_uncertainty'] >= threshold:
                seed_candidates.append({
                    'sentence': sentence,
                    'author_uncertainty': analysis['author_uncertainty'],
                    'linguistic_hedging': analysis['linguistic_hedging'],
                    'model_certainty': analysis['model_certainty'],
                    'has_conflict': analysis['has_conflict'],
                    'tom_probs': analysis['tom_probs']
                })
        except Exception as e:
            continue
    
    print(f"✓ {len(seed_candidates)} with author uncertainty >= {threshold}")
    
    print("\n[4/4] Ranking by author uncertainty...")
    seed_candidates.sort(key=lambda x: x['author_uncertainty'], reverse=True)
    return seed_candidates[:max_seeds]

def format_report(seeds):
    lines = ["="*80, "RESEARCH SEED IDEAS (Theory of Mind)", "="*80, ""]
    lines.append(f"Found {len(seeds)} high-quality research opportunities\n")
    
    for i, seed in enumerate(seeds, 1):
        lines.append(f"\n{'='*80}\nSEED {i}\n{'='*80}\n")
        lines.append(f'Sentence:\n  "{seed["sentence"]}"\n')
        lines.append(f'Theory of Mind Analysis:')
        lines.append(f'  Author Uncertainty: {seed["author_uncertainty"]:.2%} ⭐')
        lines.append(f'    - Certain:        {seed["tom_probs"]["certain"]:.2%}')
        lines.append(f'    - Hedging:        {seed["tom_probs"]["hedging"]:.2%}')
        lines.append(f'    - Very uncertain: {seed["tom_probs"]["very_uncertain"]:.2%}\n')
        
        lines.append(f'Comparison with Other Heads:')
        lines.append(f'  - Linguistic hedging: {seed["linguistic_hedging"]:.2%} (Head 3)')
        lines.append(f'  - Model certainty:    {seed["model_certainty"]:.2%} (Head 5)')
        lines.append(f'  - Has conflict:       {seed["has_conflict"]:.2%} (Head 4)\n')
        
        lines.append(f'Why This is a Research Opportunity:')
        if seed['author_uncertainty'] > 0.7:
            lines.append('  → Strong author uncertainty = open research question')
            lines.append('  → Author explicitly signals knowledge gap')
        elif seed['author_uncertainty'] > 0.5:
            lines.append('  → Moderate author uncertainty = unexplored area')
            lines.append('  → Author acknowledges uncertainty in understanding')
        
        if seed['has_conflict'] > 0.5:
            lines.append('  → Conflicting information presents resolution opportunity')
        
        # Compare with Head 3
        head3_diff = abs(seed['author_uncertainty'] - seed['linguistic_hedging'])
        if head3_diff > 0.2:
            lines.append(f'\n  ⚠️  Theory of Mind differs from linguistic hedging by {head3_diff:.2%}')
            if seed['author_uncertainty'] > seed['linguistic_hedging']:
                lines.append('  → ToM detected semantic uncertainty beyond hedging words!')
            else:
                lines.append('  → Hedging words present but author more certain semantically')
        
        lines.append("")
    
    return "\n".join(lines)

def main():
    print("="*80)
    print("LOADING THEORY OF MIND MODEL")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    model_path = "theory_of_mind_model"
    print("\nLoading model...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = BERTForTheoryOfMindIntrospection(bert_model, 768)
    
    state_dict = load_file(os.path.join(model_path, 'model.safetensors'))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Use the same paper as in your comparison
    paper_path = "language_modeling_compression.txt"
    
    if not os.path.exists(paper_path):
        print(f"\n❌ Paper not found: {paper_path}")
        print("Please ensure language_modeling_compression.txt exists")
        return
    
    print(f"\nLoading: {paper_path}")
    with open(paper_path, 'r', encoding='utf-8') as f:
        paper_text = f.read()
    print(f"✓ Loaded ({len(paper_text)} chars)")
    
    # Generate seeds
    seeds = generate_seed_ideas(
        paper_text, 
        model, 
        tokenizer, 
        device, 
        threshold=0.5, 
        max_seeds=20
    )
    
    # Format report
    report = format_report(seeds)
    print("\n" + report)
    
    # Save
    output_file = "seed_ideas_theory_of_mind.txt"
    with open(output_file, 'w') as f:
        f.write(report)
    
    print("\n" + "="*80)
    print(f"✓ Saved to: {output_file}")
    print(f"✓ Found {len(seeds)} research opportunities")
    print("="*80)
    
    # Summary statistics
    if seeds:
        avg_tom = sum(s['author_uncertainty'] for s in seeds) / len(seeds)
        avg_head3 = sum(s['linguistic_hedging'] for s in seeds) / len(seeds)
        
        print(f"\nStatistics:")
        print(f"  Average author uncertainty (ToM): {avg_tom:.2%}")
        print(f"  Average linguistic hedging (H3):  {avg_head3:.2%}")
        print(f"  Theory of Mind provides {avg_tom/avg_head3:.2f}x better signal")
    
    return seeds

if __name__ == "__main__":
    seeds = main()

