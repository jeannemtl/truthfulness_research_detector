import torch
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments, DataCollatorWithPadding  # Hugging Face transformers
from datasets import load_dataset, Features, Value, concatenate_datasets
from torch import nn 
import glob

# Define dual classifier that processes BERT's hidden states
class DualTruthfulnessClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels=2):
        super(DualTruthfulnessClassifier, self).__init__()
        self.token_classifier = nn.Linear(hidden_size, num_labels)  # Classifier for each token
        self.sentence_classifier = nn.Linear(hidden_size, num_labels)  # Classifier for whole sentence

    def forward(self, hidden_states):
        token_logits = self.token_classifier(hidden_states)  # Get token-level predictions
        sentence_logits = self.sentence_classifier(hidden_states[:, 0, :])  # Use [CLS] token for sentence prediction
        return token_logits, sentence_logits

# Combine BERT and dual classifier
class BERTForDualTruthfulness(nn.Module):
    def __init__(self, bert_model, hidden_size, num_labels=2):
        super(BERTForDualTruthfulness, self).__init__()
        self.bert = bert_model  # BERT model
        self.dual_classifier = DualTruthfulnessClassifier(hidden_size, num_labels)  # Dual classifier

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.last_hidden_state  # Get BERT's hidden states
        token_logits, sentence_logits = self.dual_classifier(hidden_states)  # Pass through classifier
        
        if labels is not None:  # Training mode
            loss_fn = nn.CrossEntropyLoss()
            # Expand single label to all tokens
            token_labels = labels.unsqueeze(1).expand(-1, token_logits.size(1))
            # Calculate token-level and sentence-level losses
            token_loss = loss_fn(token_logits.reshape(-1, token_logits.size(-1)), token_labels.reshape(-1))
            sentence_loss = loss_fn(sentence_logits, labels)
            loss = token_loss + sentence_loss  # Combine losses
            return loss
        else:  # Inference mode
            return token_logits, sentence_logits

# Set up device (MPS for Mac, CUDA for NVIDIA, or CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
model = BERTForDualTruthfulness(bert_model, hidden_size=768)
model.to(device)

# Define dataset features
features = Features({
    'statement': Value('string'),
    'label': Value('int32')
})

# Load and combine all CSV datasets
csv_files = glob.glob('publicDataset/*.csv')
datasets = [load_dataset('csv', data_files=file, features=features)['train'] for file in csv_files]
combined_dataset = concatenate_datasets(datasets)

# Tokenization function for dataset
def tokenize_function(examples):
    tokenized = tokenizer(examples['statement'], padding='max_length', truncation=True, max_length=128)
    tokenized['labels'] = examples['label']
    return tokenized

# Process dataset
tokenized_dataset = combined_dataset.map(tokenize_function, batched=True, remove_columns=combined_dataset.column_names)
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Custom data collator for batch processing
class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        batch['labels'] = torch.tensor([f['labels'] for f in features], dtype=torch.long)
        return batch

data_collator = CustomDataCollator(tokenizer=tokenizer)

# Training configuration
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Custom trainer for handling dual loss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        return (outputs, None) if return_outputs else outputs

# Initialize trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Train and save model
trainer.train()
trainer.save_model("./saved_model3")
tokenizer.save_pretrained("./saved_model3")

# Inference functions
def process_input(prompt):
    # Tokenize input and get predictions
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        token_logits, sentence_logits = model(**inputs)
    return token_logits, sentence_logits

def evaluate_truthfulness(token_logits, sentence_logits):
    # Convert logits to probabilities
    token_scores = torch.softmax(token_logits, dim=-1)[:, :, 1]
    sentence_score = torch.softmax(sentence_logits, dim=-1)[:, 1]
    return token_scores, sentence_score

def map_scores_to_tokens(prompt, token_scores, sentence_score):
    # Map scores to individual tokens for visualization
    tokens = tokenizer.tokenize(prompt)
    token_scores = token_scores.squeeze().tolist()
    print(f"Overall sentence truthfulness score: {sentence_score.item():.4f}")
    print("Token-level truthfulness scores:")
    for token, score in zip(tokens, token_scores):
        print(f"Token: {token}, Truthfulness score: {score:.4f}")

# Example usage
prompt = "The earth is flat."
token_logits, sentence_logits = process_input(prompt)
token_scores, sentence_score = evaluate_truthfulness(token_logits, sentence_logits)
map_scores_to_tokens(prompt, token_scores, sentence_score)
