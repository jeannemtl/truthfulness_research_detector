"""
Compare Head 3 (Linguistic Hedging) vs Head 6 (Theory of Mind)
Shows why Theory of Mind is superior
"""

import torch
from transformers import BertTokenizer, BertModel
from safetensors.torch import load_file
import os

class BERTForTheoryOfMindIntrospection(torch.nn.Module):
    def __init__(self, bert_model, hidden_size=768):
        super().__init__()
        self.bert = bert_model
        self.token_classifier = torch.nn.Linear(hidden_size, 2)
        self.sentence_classifier = torch.nn.Linear(hidden_size, 2)
        self.uncertainty_classifier = torch.nn.Linear(hidden_size, 3)
        self.conflict_classifier = torch.nn.Linear(hidden_size, 2)
        self.epistemic_classifier = torch.nn.Linear(hidden_size, 2)
        self.theory_of_mind_classifier = torch.nn.Linear(hidden_size, 3)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, 
                          token_type_ids=token_type_ids)
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

def compare_heads(statement, model, tokenizer, device):
    """Compare Head 3 vs Head 6 on a statement"""
    
    inputs = tokenizer(statement, return_tensors='pt', padding=True, 
                      truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Head 3: Linguistic hedging
    head3_probs = torch.softmax(outputs[2], dim=-1)[0]
    head3_hedging = head3_probs[1].item() + head3_probs[2].item()
    
    # Head 6: Theory of Mind
    head6_probs = torch.softmax(outputs[5], dim=-1)[0]
    head6_uncertainty = head6_probs[1].item() + head6_probs[2].item()
    
    return {
        'head3_hedging': head3_hedging,
        'head3_probs': head3_probs,
        'head6_uncertainty': head6_uncertainty,
        'head6_probs': head6_probs
    }

def main():
    print("="*80)
    print("COMPARING HEAD 3 (LINGUISTIC) vs HEAD 6 (THEORY OF MIND)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_path = "theory_of_mind_model"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = BERTForTheoryOfMindIntrospection(bert_model, hidden_size=768)
    
    state_dict = load_file(os.path.join(model_path, 'model.safetensors'))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Test cases showing differences
    test_cases = [
        {
            'statement': "This might explain the results",
            'expected': "Both should detect (has 'might' + author uncertain)",
        },
        {
            'statement': "This obviously involves quantum consciousness",
            'expected': "Head 3 misses (no hedging word), Head 6 detects (claim is uncertain)",
        },
        {
            'statement': "It might be that 2+2 equals 4",
            'expected': "Both detect (has 'might' + hedging about certainty)",
        },
        {
            'statement': "Water is H2O",
            'expected': "Neither should detect (fact + certain)",
        },
        {
            'statement': "The data clearly show telekinesis works",
            'expected': "Head 3 misses ('clearly' is certain word), Head 6 detects (claim is dubious)",
        },
    ]
    
    print("\nTest Cases:")
    print("="*80)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['statement']}")
        print(f"   Expected: {case['expected']}")
        print("-"*80)
        
        result = compare_heads(case['statement'], model, tokenizer, device)
        
        print(f"   HEAD 3 (Linguistic Hedging):  {result['head3_hedging']:.2%}")
        print(f"     - None:   {result['head3_probs'][0]:.3f}")
        print(f"     - Weak:   {result['head3_probs'][1]:.3f}")
        print(f"     - Strong: {result['head3_probs'][2]:.3f}")
        
        print(f"   HEAD 6 (Theory of Mind):      {result['head6_uncertainty']:.2%}")
        print(f"     - Certain:        {result['head6_probs'][0]:.3f}")
        print(f"     - Hedging:        {result['head6_probs'][1]:.3f}")
        print(f"     - Very uncertain: {result['head6_probs'][2]:.3f}")
        
        # Analysis
        diff = abs(result['head6_uncertainty'] - result['head3_hedging'])
        if diff > 0.2:
            print(f"\n   ⚠️  SIGNIFICANT DIFFERENCE: {diff:.2%}")
            if result['head6_uncertainty'] > result['head3_hedging']:
                print(f"   → Theory of Mind detected uncertainty that linguistic analysis missed!")
            else:
                print(f"   → Linguistic pattern without semantic uncertainty")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("\nHead 3 (Linguistic Hedging):")
    print("  ✓ Detects hedging WORDS ('might', 'could', 'possibly')")
    print("  ✗ Misses uncertain claims without hedging words")
    print("  ✗ Pattern matching, not semantic understanding")
    
    print("\nHead 6 (Theory of Mind):")
    print("  ✓ Detects AUTHOR'S epistemic state")
    print("  ✓ Catches uncertain claims even without hedging words")
    print("  ✓ Semantic understanding of uncertainty")
    print("  ⭐ Better for research idea generation!")

if __name__ == "__main__":
    main()

