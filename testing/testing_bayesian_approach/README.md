# Bayesian Academic Paper Novelty Analyzer

A Flask-based API that uses Bayesian uncertainty quantification to identify novel sentences in academic papers with statistically rigorous confidence estimates.

## Overview

Traditional AI models give predictions without expressing uncertainty - like a student who always claims "I'm 100% sure!" This system implements proper Bayesian methods to provide honest uncertainty estimates, transforming overconfident predictions into scientifically grounded assessments with confidence intervals.

## Key Innovation: Bayesian Uncertainty Quantification

### The Problem with Traditional Approaches
Most AI systems calculate uncertainty as simply `1 - prediction_score`. If a model predicts 70% truthfulness, it reports 30% "uncertainty" - but this isn't meaningful uncertainty, just an inverted score.

### Our Bayesian Solution
We use **Monte Carlo Dropout** to run the model multiple times with random neurons disabled, measuring how much predictions vary:

- **High Confidence**: 50 predictions are [0.68, 0.71, 0.69, 0.70, 0.72] → low variance → high confidence
- **Low Confidence**: 50 predictions are [0.45, 0.83, 0.52, 0.91, 0.38] → high variance → low confidence

This provides **epistemic uncertainty** - the model's honest assessment of its own knowledge limitations.

## Methodology

### Monte Carlo Dropout
```
For each sentence:
  Enable dropout during inference
  Run model 50 times
  Measure prediction variance
  Calculate confidence intervals
```

The model literally tells you: *"I predict 75% novelty, but I'm only confident to within ±3%"* versus *"I predict 75% novelty, but it could realistically be anywhere from 45% to 95%"*.

### Dual-Component Analysis

**Truthfulness Component**: How factually sound does the sentence appear based on learned patterns
**Relevance Component**: How well does it fit contextually with surrounding sentences

Both components use Bayesian estimation with proper uncertainty propagation.

### Conservative Ranking
Instead of ranking by raw scores, we rank by `novelty_score - uncertainty` to prioritize reliable findings. A sentence with 80% ± 2% ranks higher than 85% ± 15%.

## Statistical Foundations

### Beta Distributions for Bounded Probabilities
Since novelty scores are probabilities (0-1), we use Beta distributions rather than normal distributions, accounting for sample size effects naturally.

### Uncertainty Propagation
When combining truthfulness and relevance scores, we mathematically propagate their uncertainties:
```
Combined_uncertainty = √((uncertainty₁ × weight₁)² + (uncertainty₂ × weight₂)²)
```

### Confidence Intervals
Every prediction includes 95% confidence intervals. A result of "75% ± 5%" means we're 95% confident the true value lies between 70-80%.

## Real-World Impact

### Before: Overconfident Predictions
- Model: "This sentence is 85% novel"
- Reality: Could be anywhere from 60% to 95% novel
- Risk: Acting on unreliable information

### After: Honest Uncertainty Assessment
- Model: "This sentence is 85% ± 3% novel (high confidence)"
- or: "This sentence is 85% ± 25% novel (very uncertain, needs verification)"
- Benefit: Make informed decisions based on confidence levels

## Applications

### Academic Research
- **High Confidence Results**: Include in systematic reviews
- **Medium Confidence**: Flag for expert review
- **Low Confidence**: Exclude or gather more evidence

### Literature Discovery
- Prioritize investigating high-confidence novel findings
- Avoid wasting time on uncertain false positives
- Quantify reliability of automated literature screening

## Technical Architecture

### Model Components
- **Bayesian BERT**: Standard BERT with dropout layers for uncertainty sampling
- **Dual Classifier**: Separate heads for truthfulness and relevance prediction
- **Monte Carlo Engine**: Orchestrates multiple forward passes for uncertainty estimation
- **Statistical Layer**: Converts raw predictions into calibrated probabilities with confidence intervals

### Uncertainty Types Measured
- **Epistemic Uncertainty**: Model's knowledge limitations (reducible with more training data)
- **Prediction Variance**: How much individual predictions differ across sampling runs
- **Component Uncertainty**: Separate uncertainty estimates for truthfulness and relevance

## Scientific Validity

### Calibrated Predictions
Unlike ad-hoc uncertainty measures, our approach is grounded in Bayesian statistics and information theory. When the model says "90% confidence," it's correct approximately 90% of the time.

### Error Quantification
The system provides multiple uncertainty metrics:
- **Standard Deviation**: Spread of predictions across Monte Carlo samples
- **Confidence Width**: Range of 95% confidence interval
- **Component Breakdown**: Uncertainty attribution to different model components

## Performance Characteristics

### Computational Trade-offs
- **Speed**: 50× slower than single prediction (Monte Carlo sampling overhead)
- **Accuracy**: Dramatically improved reliability through uncertainty filtering
- **Memory**: Linear scaling with batch size and sequence length

### Quality Improvements
- **Reduced False Positives**: Low-confidence predictions filtered out
- **Better Ranking**: Conservative scoring prioritizes reliable findings
- **Interpretability**: Users understand model limitations and confidence levels

## Installation

```bash
pip install torch>=1.9.0
pip install transformers>=4.20.0
pip install safetensors>=0.3.0
pip install flask>=2.0.0
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install requests>=2.25.0
pip install PyPDF2>=3.0.0
pip install python-dotenv>=0.19.0
pip install waitress>=2.1.0
pip install anthropic>=0.3.0
```

## Understanding Output

### High Confidence Example
```json
{
  "novelty_score": 0.847,
  "uncertainty": 0.034,
  "confidence_interval": [0.780, 0.914],
  "adjusted_score": 0.813
}
```
**Interpretation**: Model is very confident this sentence is novel (uncertainty only 3.4%)

### Low Confidence Example
```json
{
  "novelty_score": 0.723,
  "uncertainty": 0.287,
  "confidence_interval": [0.150, 0.950],
  "adjusted_score": 0.436
}
```
**Interpretation**: Model is highly uncertain (28.7% uncertainty) - human verification needed

## Limitations and Scope

### What This System Provides
- Statistically rigorous uncertainty estimates
- Calibrated confidence intervals
- Conservative ranking for reliable results
- Component-level uncertainty breakdown

### What It Cannot Do
- Eliminate all prediction errors (some uncertainty is irreducible)
- Replace domain expert judgment entirely
- Process papers with complex mathematical notation perfectly
- Guarantee 100% accuracy even with high confidence

The goal is not perfect prediction, but honest, well-calibrated uncertainty that enables better human decision-making in academic research workflows.
