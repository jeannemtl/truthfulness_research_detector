"""
Improved Theory of Mind Head Training
With MUCH larger dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel
from safetensors.torch import save_file, load_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

# Same model architecture
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
        
        token_logits = self.token_classifier(hidden_states)
        sentence_logits = self.sentence_classifier(cls_hidden)
        uncertainty_logits = self.uncertainty_classifier(cls_hidden)
        conflict_logits = self.conflict_classifier(cls_hidden)
        epistemic_logits = self.epistemic_classifier(cls_hidden)
        theory_of_mind_logits = self.theory_of_mind_classifier(cls_hidden)
        
        return (token_logits, sentence_logits, uncertainty_logits, conflict_logits, 
                epistemic_logits, theory_of_mind_logits, hidden_states)

def create_large_theory_of_mind_dataset():
    """
    Create MUCH larger dataset (500+ examples)
    """
    
    # Hedging words
    weak_hedging = ["might", "could", "may", "possibly", "perhaps", "seems", "appears", 
                    "suggests", "indicates", "likely", "probably", "presumably"]
    strong_hedging = ["unclear", "uncertain", "unknown", "speculative", "inconclusive",
                     "ambiguous", "debatable", "questionable", "remains to be seen"]
    
    certain_words = ["definitely", "certainly", "clearly", "obviously", "undoubtedly",
                    "unquestionably", "absolutely", "conclusively", "proves", "demonstrates"]
    
    # Templates
    templates_uncertain = [
        "This {} explain the {}",
        "The results {} indicate a {}",
        "This {} account for the {}",
        "The {} {} suggest a {}",
        "This {} be the {}",
        "The evidence {} support this {}",
        "This {} represent a {}",
        "The {} {} reveal a {}",
    ]
    
    templates_certain = [
        "This {} explains the {}",
        "The results {} indicate a {}",
        "This {} accounts for the {}",
        "The {} {} suggest a {}",
        "This {} is the {}",
        "The evidence {} supports this {}",
        "This {} represents a {}",
        "The {} {} reveals a {}",
    ]
    
    nouns = ["phenomenon", "pattern", "data", "outcome", "mechanism", "relationship",
             "effect", "process", "structure", "system", "behavior", "observation",
             "finding", "result", "trend", "correlation", "variation", "change"]
    
    adjectives = ["experimental", "theoretical", "empirical", "statistical", "causal",
                  "significant", "notable", "important", "key", "main", "primary"]
    
    data = []
    
    # Generate weak hedging examples (label=1)
    for template in templates_uncertain:
        for hedge in weak_hedging[:6]:  # Use subset
            for noun in nouns[:10]:
                try:
                    if template.count("{}") == 2:
                        statement = template.format(hedge, noun)
                    elif template.count("{}") == 3:
                        statement = template.format(adjectives[0], hedge, noun)
                    data.append((statement, 1, "weak_hedging"))
                except:
                    pass
    
    # Generate strong hedging examples (label=2)
    strong_templates = [
        "It is {} whether this holds",
        "The relationship is {}",
        "This remains {}",
        "The mechanism is {}",
        "The evidence is {}",
        "This is {} at this point",
    ]
    
    for template in strong_templates:
        for hedge in strong_hedging[:8]:
            statement = template.format(hedge)
            data.append((statement, 2, "strong_hedging"))
    
    # Generate certain examples (label=0)
    for template in templates_certain:
        for certain in certain_words[:6]:
            for noun in nouns[:10]:
                try:
                    if template.count("{}") == 2:
                        statement = template.format(certain, noun)
                    elif template.count("{}") == 3:
                        statement = template.format(adjectives[0], certain, noun)
                    data.append((statement, 0, "certain"))
                except:
                    pass
    
    # Add factual statements
    facts = [
        ("Water is composed of hydrogen and oxygen", 0, "fact"),
        ("The Earth orbits the Sun", 0, "fact"),
        ("Photosynthesis produces oxygen", 0, "fact"),
        ("DNA contains genetic information", 0, "fact"),
        ("Gravity attracts masses", 0, "fact"),
        ("Light travels at constant speed", 0, "fact"),
        ("Cells are the basic unit of life", 0, "fact"),
        ("The moon reflects sunlight", 0, "fact"),
        ("Antibiotics kill bacteria", 0, "fact"),
        ("Plants convert light into energy", 0, "fact"),
    ] * 5  # Repeat to balance
    
    data.extend(facts)
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['statement', 'author_uncertainty', 'category'])
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✓ Created dataset: {len(df)} examples")
    print(f"\nLabel distribution:")
    print(df['author_uncertainty'].value_counts().sort_index())
    
    return df

class TheoryOfMindDataset(Dataset):
    def __init__(self, statements, labels, tokenizer, max_length=128):
        self.statements = statements
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.statements)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.statements[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_theory_of_mind_head():
    print("="*80)
    print("TRAINING THEORY OF MIND HEAD (IMPROVED)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load existing model
    existing_model_path = "/workspace/truthfulness_research_detector/uncertainty_model"
    
    print("\n[1/5] Loading existing model...")
    tokenizer = BertTokenizer.from_pretrained(existing_model_path)
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = BERTForTheoryOfMindIntrospection(bert_model, hidden_size=768)
    
    # Load existing weights
    existing_state = load_file(os.path.join(existing_model_path, 'model.safetensors'))
    model_state = model.state_dict()
    for key in existing_state.keys():
        if key in model_state:
            model_state[key] = existing_state[key]
    
    model.load_state_dict(model_state, strict=False)
    print("✓ Loaded existing weights for Heads 1-5")
    
    model.to(device)
    
    # Freeze all except Theory of Mind head
    for name, param in model.named_parameters():
        param.requires_grad = 'theory_of_mind_classifier' in name
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    
    # Create large dataset
    print("\n[2/5] Creating large dataset...")
    df = create_large_theory_of_mind_dataset()
    
    # Split
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, 
                                        stratify=df['author_uncertainty'])
    
    train_dataset = TheoryOfMindDataset(train_df['statement'].values, 
                                       train_df['author_uncertainty'].values, tokenizer)
    val_dataset = TheoryOfMindDataset(val_df['statement'].values,
                                     val_df['author_uncertainty'].values, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    print(f"✓ Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Training
    print("\n[3/5] Training...")
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 20
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            theory_of_mind_logits = outputs[5]
            
            loss = criterion(theory_of_mind_logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(theory_of_mind_logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{train_correct/train_total:.4f}'})
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                preds = torch.argmax(outputs[5], dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Acc: {train_correct/train_total:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = "theory_of_mind_model"
            os.makedirs(save_path, exist_ok=True)
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            save_file(state_dict, os.path.join(save_path, 'model.safetensors'))
            tokenizer.save_pretrained(save_path)
            print(f"  ✓ Saved (val_acc: {val_acc:.4f})")
    
    print(f"\n✓ Best validation accuracy: {best_val_acc:.4f}")
    return model, best_val_acc

if __name__ == "__main__":
    model, val_acc = train_theory_of_mind_head()
    
    print("\n" + "="*80)
    print(f"✓ Training complete! Best val acc: {val_acc:.1%}")
    print("Model saved to: theory_of_mind_model/")

