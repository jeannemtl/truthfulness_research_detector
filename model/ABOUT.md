# BERT Dual Truthfulness Classifier

## Model Architecture

### Core Components

1. **Base Model**: BERT (bert-base-uncased)
   - Generates contextual embeddings for input text
   - Hidden size: 768 dimensions

2. **Dual Classification Heads**:
   - **Token-level classifier**: Linear layer (768 → 2) analyzing each token
   - **Sentence-level classifier**: Linear layer (768 → 2) analyzing [CLS] token

### Data Flow

```
Input Text → BERT → Hidden States → ┌─ Token Classifier → Token Scores
                                    └─ Sentence Classifier → Sentence Score
```

## What It Does

This model detects truthfulness in text at two granularities:

- **Sentence-level**: Binary classification (true/false) of entire statements
- **Token-level**: Identifies which specific words contribute to truthfulness/falsehood

## Training Process

1. **Dataset**: CSV files containing statements with true/false labels
2. **Loss Function**: Combined loss from both classifiers
3. **Output**: Probability scores (0-1) where:
   - 0 = False
   - 1 = True
   - 0.5 = Uncertain

## Use Case

This model is used to analyze scientific papers by:
- Converting truthfulness scores to uncertainty metrics
- Identifying claims and statements made in an arxiv apper that may warrant further investigation
- Generating research seed ideas from uncertain/ambiguous statements that can be populated within a scientific agent like SAKANA AI Scientist. Read the HOWTO.md.

## Relevance and Uncertainty Calculation

**Relevance Calculation**: After obtaining sentence embeddings from BERT's [CLS] token, cosine similarity is computed between each sentence and a research query embedding (e.g., "LLM falsehood detection"). This produces a score from 0-1 indicating topical relevance.

**Uncertainty Derivation**: The truthfulness probability from the sentence classifier is transformed into an uncertainty metric using the formula: `uncertainty = 1 - |truthfulness_score - 0.5| × 2`. This yields maximum uncertainty (1.0) when the model outputs 0.5 (cannot determine true/false), and minimum uncertainty (0.0) for confident predictions near 0 or 1. 

**Combined Scoring**: Sentences with high uncertainty and high relevance scores are selected as potential novel research directions, effectively identifying scientifically relevant claims that the model finds ambiguous or disputed.
