"""
Research Seed Idea Generation using Theory of Mind Head

Uses Head 6 (Theory of Mind) to detect author uncertainty,
which indicates genuine research opportunities.
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from safetensors.torch import load_file
import os
import re
from typing import List, Dict, Tuple

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class BERTForTheoryOfMindIntrospection(nn.Module):
    """6-head model with Theory of Mind head"""
    
    def __init__(self, bert_model, hidden_size=768):
        super().__init__()
        self.bert = bert_model
        
        self.token_classifier = nn.Linear(hidden_size, 2)
        self.sentence_classifier = nn.Linear(hidden_size, 2)
        self.uncertainty_classifier = nn.Linear(hidden_size, 3)
        self.conflict_classifier = nn.Linear(hidden_size, 2)
        self.epistemic_classifier = nn.Linear(hidden_size, 2)
        self.theory_of_mind_classifier = nn.Linear(hidden_size, 3)  # NEW!
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        hidden_states = outputs.last_hidden_state
        cls_hidden = hidden_states[:, 0, :]
        
        token_logits = self.token_classifier(hidden_states)
        sentence_logits = self.sentence_classifier(cls_hidden)
        uncertainty_logits = self.uncertainty_classifier(cls_hidden)
        conflict_logits = self.conflict_classifier(cls_hidden)
        epistemic_logits = self.epistemic_classifier(cls_hidden)
        theory_of_mind_logits = self.theory_of_mind_classifier(cls_hidden)
        
        return (
            token_logits,
            sentence_logits,
            uncertainty_logits,
            conflict_logits,
            epistemic_logits,
            theory_of_mind_logits,  # Head 6
            hidden_states
        )

# =============================================================================
# PAPER PROCESSING
# =============================================================================

def extract_sentences(text: str) -> List[str]:
    """Extract sentences from paper text"""
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    
    # Clean and filter
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [s for s in sentences if len(s.split()) >= 5]  # At least 5 words
    sentences = [s for s in sentences if len(s.split()) <= 50]  # Max 50 words
    
    return sentences

def is_research_relevant(sentence: str) -> bool:
    """Filter for research-relevant sentences"""
    
    # Skip references, citations, metadata
    skip_patterns = [
        r'^\d+\.',  # Numbered lists
        r'Figure \d+',
        r'Table \d+',
        r'Section \d+',
        r'\[[\d,\s]+\]',  # Citations
        r'et al\.',
        r'@',
        r'http',
    ]
    
    for pattern in skip_patterns:
        if re.search(pattern, sentence):
            return False
    
    # Require substantive content
    if len(sentence.split()) < 10:
        return False
    
    return True

# =============================================================================
# THEORY OF MIND ANALYSIS
# =============================================================================

def detect_author_uncertainty(
    statement: str,
    model: BERTForTheoryOfMindIntrospection,
    tokenizer: BertTokenizer,
    device: torch.device
) -> Tuple[float, torch.Tensor]:
    """
    Detect author's uncertainty using Theory of Mind head
    
    Returns:
        author_uncertainty: float [0-1]
        probs: [P(certain), P(hedging), P(very_uncertain)]
    """
    inputs = tokenizer(
        statement,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        theory_of_mind_logits = outputs[5]  # Head 6
    
    probs = torch.softmax(theory_of_mind_logits, dim=-1)[0]
    
    # Author uncertainty = hedging + very_uncertain
    author_uncertainty = probs[1].item() + probs[2].item()
    
    return author_uncertainty, probs

def analyze_comprehensive(
    statement: str,
    model: BERTForTheoryOfMindIntrospection,
    tokenizer: BertTokenizer,
    device: torch.device
) -> Dict:
    """
    Comprehensive analysis using all 6 heads
    """
    inputs = tokenizer(
        statement,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Head 3: Linguistic hedging
    uncertainty_probs = torch.softmax(outputs[2], dim=-1)[0]
    linguistic_hedging = uncertainty_probs[1].item() + uncertainty_probs[2].item()
    
    # Head 4: Conflict detection
    conflict_probs = torch.softmax(outputs[3], dim=-1)[0]
    has_conflict = conflict_probs[1].item()
    
    # Head 5: Model's epistemic certainty
    epistemic_probs = torch.softmax(outputs[4], dim=-1)[0]
    model_certainty = epistemic_probs[0].item()
    
    # Head 6: Theory of Mind (author's uncertainty)
    tom_probs = torch.softmax(outputs[5], dim=-1)[0]
    author_uncertainty = tom_probs[1].item() + tom_probs[2].item()
    
    return {
        'linguistic_hedging': linguistic_hedging,      # Head 3
        'has_conflict': has_conflict,                  # Head 4
        'model_certainty': model_certainty,            # Head 5
        'author_uncertainty': author_uncertainty,       # Head 6 (Theory of Mind)
        'tom_probs': {
            'certain': tom_probs[0].item(),
            'hedging': tom_probs[1].item(),
            'very_uncertain': tom_probs[2].item()
        }
    }

# =============================================================================
# SEED IDEA GENERATION
# =============================================================================

def generate_seed_ideas(
    paper_text: str,
    model: BERTForTheoryOfMindIntrospection,
    tokenizer: BertTokenizer,
    device: torch.device,
    threshold: float = 0.5,
    max_seeds: int = 20
) -> List[Dict]:
    """
    Generate seed ideas from paper using Theory of Mind
    
    Args:
        paper_text: Full paper text
        model: Theory of Mind model
        tokenizer: BERT tokenizer
        device: torch device
        threshold: Author uncertainty threshold (default 0.5)
        max_seeds: Maximum number of seeds to return
    
    Returns:
        List of seed candidates with analysis
    """
    
    print("="*80)
    print("SEED IDEA GENERATION WITH THEORY OF MIND")
    print("="*80)
    
    # Extract sentences
    print("\n[1/4] Extracting sentences from paper...")
    sentences = extract_sentences(paper_text)
    print(f"✓ Extracted {len(sentences)} sentences")
    
    # Filter for research-relevant
    print("\n[2/4] Filtering for research-relevant sentences...")
    relevant = [s for s in sentences if is_research_relevant(s)]
    print(f"✓ {len(relevant)} research-relevant sentences")
    
    # Analyze with Theory of Mind
    print("\n[3/4] Analyzing author uncertainty (Theory of Mind)...")
    seed_candidates = []
    
    for sentence in relevant:
        analysis = analyze_comprehensive(sentence, model, tokenizer, device)
        
        # Only keep if author shows uncertainty
        if analysis['author_uncertainty'] >= threshold:
            seed_candidates.append({
                'sentence': sentence,
                'author_uncertainty': analysis['author_uncertainty'],
                'linguistic_hedging': analysis['linguistic_hedging'],
                'model_certainty': analysis['model_certainty'],
                'has_conflict': analysis['has_conflict'],
                'tom_probs': analysis['tom_probs']
            })
    
    print(f"✓ Found {len(seed_candidates)} sentences with author uncertainty")
    
    # Sort by author uncertainty (highest first)
    print("\n[4/4] Ranking by author uncertainty...")
    seed_candidates.sort(key=lambda x: x['author_uncertainty'], reverse=True)
    
    # Return top N
    top_seeds = seed_candidates[:max_seeds]
    
    print(f"✓ Returning top {len(top_seeds)} seed candidates")
    
    return top_seeds

def format_seed_report(seeds: List[Dict]) -> str:
    """Format seed ideas as a report"""
    
    report = []
    report.append("="*80)
    report.append("RESEARCH SEED IDEAS (Theory of Mind)")
    report.append("="*80)
    report.append("")
    
    for i, seed in enumerate(seeds, 1):
        report.append(f"\n{'='*80}")
        report.append(f"SEED {i}")
        report.append(f"{'='*80}")
        report.append("")
        
        # The sentence
        report.append(f"Sentence:")
        report.append(f"  \"{seed['sentence']}\"")
        report.append("")
        
        # Theory of Mind analysis
        report.append(f"Theory of Mind Analysis:")
        report.append(f"  Author Uncertainty: {seed['author_uncertainty']:.2%} ⭐")
        report.append(f"    - Certain:        {seed['tom_probs']['certain']:.2%}")
        report.append(f"    - Hedging:        {seed['tom_probs']['hedging']:.2%}")
        report.append(f"    - Very uncertain: {seed['tom_probs']['very_uncertain']:.2%}")
        report.append("")
        
        # Comparison with other heads
        report.append(f"Comparison:")
        report.append(f"  Linguistic hedging: {seed['linguistic_hedging']:.2%} (Head 3)")
        report.append(f"  Model certainty:    {seed['model_certainty']:.2%} (Head 5)")
        report.append(f"  Has conflict:       {seed['has_conflict']:.2%} (Head 4)")
        report.append("")
        
        # Research opportunity explanation
        report.append(f"Why This is a Research Opportunity:")
        if seed['author_uncertainty'] > 0.7:
            report.append(f"  → Strong author uncertainty indicates open question")
        elif seed['author_uncertainty'] > 0.5:
            report.append(f"  → Moderate author uncertainty suggests unexplored area")
        
        if seed['has_conflict'] > 0.5:
            report.append(f"  → Conflicting information presents resolution opportunity")
        
        report.append("")
    
    return "\n".join(report)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution"""
    
    print("="*80)
    print("LOADING THEORY OF MIND MODEL")
    print("="*80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model
    model_path = "theory_of_mind_model"
    
    print("\nLoading model...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = BERTForTheoryOfMindIntrospection(bert_model, hidden_size=768)
    
    state_dict = load_file(os.path.join(model_path, 'model.safetensors'))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully")
    
    # Load paper
    paper_path = "/mnt/project/INTROSPECTION_paer"
    
    print(f"\nLoading paper: {paper_path}")
    with open(paper_path, 'r', encoding='utf-8') as f:
        paper_text = f.read()
    
    print(f"✓ Loaded paper ({len(paper_text)} characters)")
    
    # Generate seeds
    seeds = generate_seed_ideas(
        paper_text=paper_text,
        model=model,
        tokenizer=tokenizer,
        device=device,
        threshold=0.5,
        max_seeds=20
    )
    
    # Format and print report
    report = format_seed_report(seeds)
    print("\n" + report)
    
    # Save to file
    output_file = "seed_ideas_theory_of_mind.txt"
    with open(output_file, 'w') as f:
        f.write(report)
    
    print("\n" + "="*80)
    print(f"✓ Saved to: {output_file}")
    print("="*80)
    
    return seeds

if __name__ == "__main__":
    seeds = main()

