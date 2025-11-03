"""
Introspective Uncertainty Classifier
Based on "Emergent Introspective Awareness in Large Language Models" (Anthropic, 2025)

This model distinguishes:
1. Objective truth value (true/false)
2. Author's uncertainty communication (confident/hedged/explicit)
3. Introspective conflict (mismatch between truth and expressed certainty)
"""

import torch
from torch import nn
from transformers import BertModel, BertConfig


class IntrospectiveUncertaintyClassifier(nn.Module):
    """
    Multi-task classifier that processes BERT's hidden states to detect:
    - Token-level truthfulness
    - Sentence-level truthfulness
    - Uncertainty communication patterns
    - Introspective conflicts
    """
    
    def __init__(self, hidden_size, num_labels=2):
        super().__init__()
        
        # Original truthfulness classifiers
        self.token_classifier = nn.Linear(hidden_size, num_labels)
        self.sentence_classifier = nn.Linear(hidden_size, num_labels)
        
        # Uncertainty communication classifier
        self.uncertainty_classifier = nn.Linear(hidden_size, 3)
        # 3 classes: [confident, hedged, explicit_uncertainty]
        
        # Introspective conflict detector
        # Detects when truth value conflicts with certainty expression
        self.conflict_detector = nn.Linear(hidden_size, 2)
        # 2 classes: [consistent, conflicted]
        
        # Optional: Epistemic state classifier (author's belief vs objective truth)
        self.epistemic_classifier = nn.Linear(hidden_size, 2)
        # 2 classes: [author_confident, author_uncertain]

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: BERT's last hidden state [batch_size, seq_len, hidden_size]
        
        Returns:
            tuple of logits for each classification task
        """
        # Token-level predictions (all tokens)
        token_logits = self.token_classifier(hidden_states)
        
        # Sentence-level predictions (CLS token only)
        cls_hidden = hidden_states[:, 0, :]
        sentence_logits = self.sentence_classifier(cls_hidden)
        
        # Uncertainty communication detection
        uncertainty_logits = self.uncertainty_classifier(cls_hidden)
        
        # Introspective conflict detection
        conflict_logits = self.conflict_detector(cls_hidden)
        
        # Epistemic state
        epistemic_logits = self.epistemic_classifier(cls_hidden)
        
        return (token_logits, sentence_logits, uncertainty_logits, 
                conflict_logits, epistemic_logits)


class BERTForIntrospectiveUncertainty(nn.Module):
    """
    Complete model combining BERT encoder with introspective classifiers
    """
    
    def __init__(self, bert_model, hidden_size, num_labels=2):
        super().__init__()
        self.bert = bert_model
        self.classifier = IntrospectiveUncertaintyClassifier(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, 
                truth_labels=None, uncertainty_labels=None, 
                conflict_labels=None, epistemic_labels=None):
        """
        Forward pass with optional training labels
        
        Args:
            input_ids: Token indices [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs (optional)
            truth_labels: Ground truth labels for truthfulness [batch_size]
            uncertainty_labels: Labels for uncertainty type [batch_size]
            conflict_labels: Labels for introspective conflicts [batch_size]
            epistemic_labels: Labels for epistemic state [batch_size]
        
        Returns:
            If training (labels provided): total_loss
            If inference: tuple of (all_logits, hidden_states)
        """
        # Get BERT representations
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        hidden_states = outputs.last_hidden_state
        
        # Get predictions from classifiers
        (token_logits, sentence_logits, uncertainty_logits, 
         conflict_logits, epistemic_logits) = self.classifier(hidden_states)
        
        # Training mode
        if truth_labels is not None:
            return self._compute_training_loss(
                token_logits, sentence_logits, uncertainty_logits,
                conflict_logits, epistemic_logits, hidden_states,
                truth_labels, uncertainty_labels, conflict_labels, epistemic_labels
            )
        
        # Inference mode
        return (token_logits, sentence_logits, uncertainty_logits, 
                conflict_logits, epistemic_logits, hidden_states)
    
    def _compute_training_loss(self, token_logits, sentence_logits, 
                               uncertainty_logits, conflict_logits, epistemic_logits,
                               hidden_states, truth_labels, uncertainty_labels,
                               conflict_labels, epistemic_labels):
        """
        Compute multi-task training loss
        """
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        
        # 1. Truthfulness loss (token + sentence level)
        if truth_labels is not None:
            # Token-level: expand labels to all positions
            token_labels = truth_labels.unsqueeze(1).expand(-1, token_logits.size(1))
            token_loss = loss_fn(
                token_logits.reshape(-1, token_logits.size(-1)), 
                token_labels.reshape(-1)
            )
            # Sentence-level
            sentence_loss = loss_fn(sentence_logits, truth_labels)
            truth_loss = token_loss + sentence_loss
            total_loss += truth_loss
        
        # 2. Uncertainty communication loss
        if uncertainty_labels is not None:
            uncertainty_loss = loss_fn(uncertainty_logits, uncertainty_labels)
            total_loss += 0.5 * uncertainty_loss  # Weight = 0.5
        
        # 3. Introspective conflict loss
        if conflict_labels is not None:
            conflict_loss = loss_fn(conflict_logits, conflict_labels)
            total_loss += 0.3 * conflict_loss  # Weight = 0.3
        elif uncertainty_labels is not None and truth_labels is not None:
            # Auto-generate conflict labels if not provided
            # Conflict = (true AND hedged) OR (false AND confident)
            auto_conflict = ((truth_labels == 1) & (uncertainty_labels > 0)) | \
                           ((truth_labels == 0) & (uncertainty_labels == 0))
            conflict_loss = loss_fn(conflict_logits, auto_conflict.long())
            total_loss += 0.3 * conflict_loss
        
        # 4. Epistemic state loss
        if epistemic_labels is not None:
            epistemic_loss = loss_fn(epistemic_logits, epistemic_labels)
            total_loss += 0.2 * epistemic_loss  # Weight = 0.2
        
        return total_loss

    @classmethod
    def from_pretrained(cls, model_path, config=None):
        """
        Load model from pretrained checkpoint
        
        Args:
            model_path: Path to saved model directory
            config: Optional BertConfig (will load from model_path if not provided)
        """
        if config is None:
            config = BertConfig.from_pretrained(model_path)
        
        bert_model = BertModel.from_pretrained(model_path, config=config)
        model = cls(bert_model, hidden_size=config.hidden_size)
        
        return model


# Uncertainty marker detection (linguistic features)
UNCERTAINTY_MARKERS = {
    'hedge_words': [
        'may', 'might', 'could', 'possibly', 'perhaps', 'likely', 'unlikely',
        'probably', 'apparently', 'seemingly', 'arguably', 'presumably',
        'conceivably', 'potentially', 'plausibly'
    ],
    'explicit_uncertainty': [
        'unclear', 'uncertain', 'unknown', 'inconclusive', 'ambiguous',
        'debatable', 'questionable', 'disputed', 'controversial',
        'indeterminate', 'unresolved', 'unverified'
    ],
    'epistemic_qualifiers': [
        'appears to', 'seems to', 'suggests', 'indicates', 'implies',
        'tends to', 'is thought to', 'is believed to', 'is considered',
        'is assumed', 'is supposed', 'purportedly', 'allegedly'
    ],
    'research_qualifiers': [
        'preliminary', 'tentative', 'exploratory', 'speculative',
        'hypothetical', 'theoretical', 'proposed', 'suggested',
        'further research needed', 'requires investigation',
        'needs validation', 'to be confirmed'
    ]
}


def detect_linguistic_uncertainty_markers(text):
    """
    Detect uncertainty markers in text using pattern matching
    
    Args:
        text: Input text string
    
    Returns:
        dict with marker categories and found markers
    """
    text_lower = text.lower()
    found_markers = {}
    
    for category, markers in UNCERTAINTY_MARKERS.items():
        found = [marker for marker in markers if marker in text_lower]
        if found:
            found_markers[category] = found
    
    return found_markers
