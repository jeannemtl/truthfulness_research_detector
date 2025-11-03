# Training Theory of Mind Head

## What This Does

Trains a **NEW Head 6** that explicitly detects **AUTHOR'S uncertainty**, separate from:
- Head 3: Hedging detection (linguistic patterns)
- Head 5: Epistemic certainty (model's own confidence)

## Key Innovation

**Theory of Mind Introspection:**
- Model introspects on its internal representation of AUTHOR'S mental state
- Not just pattern matching on "might", "could"
- Detects author's epistemic state from hidden layer activations

## Run Training
```bash
python train_theory_of_mind_head.py
```

## What Happens

1. **Loads existing model** (Heads 1-5)
2. **Adds new Head 6** (Theory of Mind - randomly initialized)
3. **Freezes Heads 1-5** (only train Head 6)
4. **Creates dataset** with explicit author uncertainty labels
5. **Trains for 10 epochs**
6. **Saves to:** `theory_of_mind_model/`

## Dataset

Training examples explicitly labeled for AUTHOR'S mental state:
```python
# Author uncertain (label=1)
"This might explain X"
"The results could indicate Y"
"It is unclear whether Z"

# Author certain (label=0)  
"This definitely explains X"
"The data clearly show Y"

# Tricky cases
"This obviously involves quantum consciousness"  # label=1 (claim is uncertain!)
"2+2 might equal 4"  # label=1 (author hedging despite certain fact)
```

## Expected Output
```
================================================================================
TRAINING THEORY OF MIND HEAD
================================================================================

Device: cuda

[1/5] Loading existing model...
✓ Loaded existing weights for Heads 1-5
✓ Theory of Mind head (Head 6) initialized randomly

Trainable parameters: 2,307 / 109,483,779 (0.00%)

[2/5] Creating Theory of Mind dataset...
Created Theory of Mind dataset: 60 examples

Label distribution:
0    25
1    25
2    10

[3/5] Setting up training...
✓ Train: 48 | Val: 12

[4/5] Training Theory of Mind head...
Epoch 1/10:
  Train Loss: 0.8234 | Train Acc: 0.6458
  Val Loss:   0.7112 | Val Acc:   0.7500
  ✓ Saved best model (val_acc: 0.7500)

...

Epoch 10/10:
  Train Loss: 0.1234 | Train Acc: 0.9583
  Val Loss:   0.2341 | Val Acc:   0.9167
  ✓ Saved best model (val_acc: 0.9167)

[5/5] Training complete!
Best validation accuracy: 0.9167
Model saved to: theory_of_mind_model/

================================================================================
TESTING THEORY OF MIND HEAD
================================================================================

Test Results:
--------------------------------------------------------------------------------

✓ This might be the correct explanation
   Expected: hedging        | Predicted: hedging
   Probs: certain=0.123, hedging=0.789, very_uncertain=0.088

✓ This definitely explains the phenomenon
   Expected: certain        | Predicted: certain
   Probs: certain=0.912, hedging=0.067, very_uncertain=0.021

✗ This obviously involves quantum consciousness
   Expected: hedging        | Predicted: certain
   Probs: certain=0.723, hedging=0.234, very_uncertain=0.043

================================================================================
SUMMARY
================================================================================
Validation Accuracy: 91.7%
Test Accuracy:       88.9%

Model saved to: theory_of_mind_model/

This head detects AUTHOR'S uncertainty, not model's own uncertainty!
```

## Next Steps

After training, use this head in your seed idea generation:
```python
# Load Theory of Mind model
model = BERTForTheoryOfMindIntrospection(bert_model, hidden_size=768)
state_dict = load_file("theory_of_mind_model/model.safetensors")
model.load_state_dict(state_dict)

# Detect author uncertainty
outputs = model(statement)
theory_of_mind_logits = outputs[5]  # Head 6
author_uncertainty = softmax(theory_of_mind_logits)[0][1] + [2]  # hedging + very_uncertain

# Use for idea generation
if author_uncertainty > 0.5:
    # Author was uncertain → good research opportunity!
    generate_idea(statement)
```

## Files Created

- `theory_of_mind_model/model.safetensors` - Model weights (all 6 heads)
- `theory_of_mind_model/tokenizer_config.json` - Tokenizer config
- `theory_of_mind_model/vocab.txt` - Vocabulary

