"""
Introspective Analysis Module
Provides functions for analyzing statements using the trained uncertainty classifier
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def introspective_uncertainty_analysis(
    statement: str, 
    model, 
    tokenizer, 
    device
) -> Dict:
    """
    Perform introspective analysis on a single statement
    
    Distinguishes:
    1. Objective truth value (is the statement factually true?)
    2. Author's uncertainty communication (is uncertainty being expressed?)
    3. Introspective conflict (mismatch between truth and expressed certainty)
    
    Args:
        statement: Text to analyze
        model: Trained BERTForIntrospectiveUncertainty model
        tokenizer: BERT tokenizer
        device: torch device
    
    Returns:
        Dictionary with detailed analysis
    """
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(
        statement, 
        return_tensors='pt', 
        truncation=True, 
        padding=True,
        max_length=128
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        (token_logits, sentence_logits, uncertainty_logits, 
         conflict_logits, epistemic_logits, hidden_states) = outputs
    
    # Convert logits to probabilities
    truth_probs = torch.softmax(sentence_logits, dim=-1)[0]
    uncertainty_probs = torch.softmax(uncertainty_logits, dim=-1)[0]
    conflict_probs = torch.softmax(conflict_logits, dim=-1)[0]
    epistemic_probs = torch.softmax(epistemic_logits, dim=-1)[0]
    
    # Token-level analysis
    token_scores = torch.softmax(token_logits, dim=-1)[:, :, 1]
    token_variance = torch.var(token_scores).item()
    token_mean = torch.mean(token_scores).item()
    
    # Classify uncertainty type
    uncertainty_idx = torch.argmax(uncertainty_probs).item()
    uncertainty_types = ['confident', 'hedged', 'explicit']
    uncertainty_type = uncertainty_types[uncertainty_idx]
    
    # Truth assessment
    truth_prob = truth_probs[1].item()
    is_likely_true = truth_prob > 0.5
    
    # Conflict detection
    conflict_prob = conflict_probs[1].item()
    is_conflicted = conflict_prob > 0.5
    
    # Interpret conflict
    conflict_interpretation = _interpret_conflict(
        truth_prob, uncertainty_type, conflict_prob
    )
    
    # Detect linguistic markers
    from model.architectures.introspective_uncertainty import detect_linguistic_uncertainty_markers
    linguistic_markers = detect_linguistic_uncertainty_markers(statement)
    
    # Assess reliability
    reliability = assess_reliability(truth_prob, uncertainty_type, conflict_prob)
    
    return {
        "statement": statement,
        
        # Objective truth assessment
        "truth_probability": truth_prob,
        "is_likely_true": is_likely_true,
        "truth_confidence": max(truth_probs).item(),
        
        # Uncertainty communication
        "uncertainty_type": uncertainty_type,
        "confidence_score": uncertainty_probs[0].item(),
        "hedging_score": uncertainty_probs[1].item(),
        "explicit_uncertainty_score": uncertainty_probs[2].item(),
        
        # Epistemic state (author's belief)
        "author_appears_confident": epistemic_probs[0].item() > 0.5,
        "author_confidence_prob": epistemic_probs[0].item(),
        "author_uncertainty_prob": epistemic_probs[1].item(),
        
        # Introspective awareness metrics
        "introspective_conflict_probability": conflict_prob,
        "is_conflicted": is_conflicted,
        "conflict_interpretation": conflict_interpretation,
        "token_variance": token_variance,
        "token_mean_truth_score": token_mean,
        
        # Linguistic features
        "linguistic_uncertainty_markers": linguistic_markers,
        "has_linguistic_markers": len(linguistic_markers) > 0,
        
        # Overall assessment
        "author_signaling_uncertainty": uncertainty_type != 'confident',
        "reliability_assessment": reliability,
        "recommendation": _generate_recommendation(
            truth_prob, uncertainty_type, conflict_prob
        )
    }


def _interpret_conflict(
    truth_prob: float, 
    uncertainty_type: str, 
    conflict_prob: float
) -> Optional[str]:
    """
    Interpret the nature of introspective conflict
    """
    if conflict_prob <= 0.5:
        return None
    
    if truth_prob > 0.7 and uncertainty_type != 'confident':
        return ("True statement presented with uncertainty markers - "
                "author may be expressing genuine epistemic caution or "
                "acknowledging limitations of evidence")
    
    elif truth_prob < 0.3 and uncertainty_type == 'confident':
        return ("False statement presented confidently - "
                "potential misinformation, error, or confident but incorrect claim")
    
    elif 0.3 <= truth_prob <= 0.7 and uncertainty_type == 'confident':
        return ("Ambiguous truth value stated confidently - "
                "author may be overconfident given uncertainty in domain")
    
    elif 0.3 <= truth_prob <= 0.7 and uncertainty_type != 'confident':
        return ("Ambiguous statement with appropriate uncertainty markers - "
                "author correctly signals uncertainty about unclear claim")
    
    return "Complex conflict pattern detected"


def assess_reliability(
    truth_prob: float, 
    uncertainty_type: str, 
    conflict_prob: float
) -> str:
    """
    Provide overall reliability assessment
    """
    if truth_prob > 0.8 and uncertainty_type == 'confident' and conflict_prob < 0.3:
        return "VERY HIGH - True statement, confidently and appropriately stated"
    
    elif truth_prob > 0.7 and uncertainty_type == 'confident':
        return "HIGH - Likely true statement stated confidently"
    
    elif truth_prob > 0.7 and uncertainty_type != 'confident':
        return "MEDIUM-HIGH - True statement with epistemic caution"
    
    elif 0.4 <= truth_prob <= 0.7 and uncertainty_type != 'confident':
        return "MEDIUM - Ambiguous claim with appropriate uncertainty"
    
    elif truth_prob < 0.3 and uncertainty_type != 'confident':
        return "MEDIUM-LOW - False claim but uncertainty acknowledged"
    
    elif truth_prob < 0.3 and uncertainty_type == 'confident':
        return "VERY LOW - Likely false statement presented as fact"
    
    else:
        return "UNCERTAIN - Complex or ambiguous assessment"


def _generate_recommendation(
    truth_prob: float, 
    uncertainty_type: str, 
    conflict_prob: float
) -> str:
    """
    Generate actionable recommendation
    """
    if truth_prob > 0.8 and uncertainty_type == 'confident':
        return "Accept as reliable claim"
    
    elif truth_prob > 0.7 and uncertainty_type != 'confident':
        return "Likely accurate but author signals uncertainty - verify if critical"
    
    elif truth_prob < 0.3 and uncertainty_type == 'confident':
        return "CAUTION: Potentially false claim stated confidently - verify before accepting"
    
    elif truth_prob < 0.3 and uncertainty_type != 'confident':
        return "False but author acknowledges uncertainty - treat as speculative"
    
    elif conflict_prob > 0.6:
        return "High introspective conflict detected - cross-reference with other sources"
    
    else:
        return "Ambiguous reliability - seek additional evidence"


def batch_analyze_statements(
    statements: List[str],
    model,
    tokenizer,
    device,
    batch_size: int = 32
) -> List[Dict]:
    """
    Analyze multiple statements efficiently in batches
    
    Args:
        statements: List of statements to analyze
        model: Trained model
        tokenizer: Tokenizer
        device: torch device
        batch_size: Batch size for processing
    
    Returns:
        List of analysis dictionaries
    """
    results = []
    
    for i in range(0, len(statements), batch_size):
        batch = statements[i:i+batch_size]
        batch_results = [
            introspective_uncertainty_analysis(stmt, model, tokenizer, device)
            for stmt in batch
        ]
        results.extend(batch_results)
    
    return results


def analyze_document_uncertainty(
    sentences: List[str],
    model,
    tokenizer,
    device,
    min_sentence_length: int = 5
) -> Dict:
    """
    Analyze uncertainty patterns across an entire document
    
    Args:
        sentences: List of sentences from document
        model: Trained model
        tokenizer: Tokenizer
        device: torch device
        min_sentence_length: Minimum words per sentence to analyze
    
    Returns:
        Dictionary with document-level analysis
    """
    # Filter short sentences
    valid_sentences = [
        s for s in sentences 
        if len(s.split()) >= min_sentence_length
    ]
    
    # Analyze all sentences
    analyses = batch_analyze_statements(valid_sentences, model, tokenizer, device)
    
    # Aggregate statistics
    uncertain_statements = [a for a in analyses if a['author_signaling_uncertainty']]
    conflicted_statements = [a for a in analyses if a['is_conflicted']]
    high_risk = [a for a in analyses if 'CAUTION' in a['recommendation']]
    
    # Calculate metrics
    total = len(analyses)
    uncertainty_ratio = len(uncertain_statements) / total if total > 0 else 0
    conflict_ratio = len(conflicted_statements) / total if total > 0 else 0
    risk_ratio = len(high_risk) / total if total > 0 else 0
    
    # Average scores
    avg_truth = np.mean([a['truth_probability'] for a in analyses])
    avg_hedging = np.mean([a['hedging_score'] for a in analyses])
    avg_conflict = np.mean([a['introspective_conflict_probability'] for a in analyses])
    
    return {
        "total_sentences_analyzed": total,
        "uncertainty_ratio": uncertainty_ratio,
        "conflict_ratio": conflict_ratio,
        "high_risk_ratio": risk_ratio,
        
        "average_scores": {
            "truth_probability": avg_truth,
            "hedging_score": avg_hedging,
            "conflict_probability": avg_conflict
        },
        
        "top_uncertain_statements": sorted(
            uncertain_statements,
            key=lambda x: x['hedging_score'] + x['explicit_uncertainty_score'],
            reverse=True
        )[:10],
        
        "conflicted_statements": sorted(
            conflicted_statements,
            key=lambda x: x['introspective_conflict_probability'],
            reverse=True
        )[:10],
        
        "high_risk_statements": high_risk[:10],
        
        "summary": _generate_document_summary(
            uncertainty_ratio, conflict_ratio, risk_ratio, avg_truth
        )
    }


def _generate_document_summary(
    uncertainty_ratio: float,
    conflict_ratio: float,
    risk_ratio: float,
    avg_truth: float
) -> str:
    """
    Generate human-readable document summary
    """
    summary = []
    
    # Uncertainty assessment
    if uncertainty_ratio > 0.3:
        summary.append(f"High uncertainty: {uncertainty_ratio:.1%} of statements contain hedging")
    elif uncertainty_ratio > 0.15:
        summary.append(f"Moderate uncertainty: {uncertainty_ratio:.1%} of statements hedged")
    else:
        summary.append(f"Low uncertainty: {uncertainty_ratio:.1%} of statements hedged")
    
    # Conflict assessment
    if conflict_ratio > 0.2:
        summary.append(f"Significant conflicts detected ({conflict_ratio:.1%})")
    
    # Risk assessment
    if risk_ratio > 0.1:
        summary.append(f"CAUTION: {risk_ratio:.1%} of statements are high-risk")
    
    # Overall truth
    if avg_truth > 0.7:
        summary.append("Overall high factual reliability")
    elif avg_truth < 0.4:
        summary.append("Overall low factual reliability")
    
    return ". ".join(summary) + "."
