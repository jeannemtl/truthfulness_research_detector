"""
Training Script for Introspective Uncertainty Classifier
Based on "Emergent Introspective Awareness in Large Language Models" principles
"""

import torch
from torch import nn
from transformers import (
    BertTokenizer, BertModel, Trainer, TrainingArguments, 
    DataCollatorWithPadding, EarlyStoppingCallback
)
from datasets import load_dataset, Features, Value, ClassLabel
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architectures.introspective_uncertainty import BERTForIntrospectiveUncertainty


class UncertaintyTrainer:
    """
    Trainer for introspective uncertainty detection model
    """
    
    def __init__(self, model_name='bert-base-uncased', output_dir='./uncertainty_model'):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = self._get_device()
        
        print(f"Using device: {self.device}")
        
    def _get_device(self):
        """Detect available device"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def load_data(self, data_path):
        """
        Load dataset with uncertainty labels
        
        Args:
            data_path: Path to CSV file or directory with train/test split
        """
        print(f"\nLoading dataset from: {data_path}")
        
        # Define dataset features
        features = Features({
            'statement': Value('string'),
            'label': Value('int32'),
            'has_uncertainty_marker': Value('int32'),
            'uncertainty_type': ClassLabel(names=['confident', 'hedged', 'explicit']),
            'has_conflict': Value('int32'),
        })
        
        # Check if we have train/test split
        train_path = data_path.replace('.csv', '_train.csv')
        test_path = data_path.replace('.csv', '_test.csv')
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            print("Found existing train/test split")
            train_dataset = load_dataset('csv', data_files=train_path, features=features)['train']
            test_dataset = load_dataset('csv', data_files=test_path, features=features)['train']
        else:
            print("Creating train/test split")
            dataset = load_dataset('csv', data_files=data_path, features=features)['train']
            split = dataset.train_test_split(test_size=0.2, seed=42)
            train_dataset = split['train']
            test_dataset = split['test']
        
        print(f"Train size: {len(train_dataset)}")
        print(f"Test size: {len(test_dataset)}")
        
        return train_dataset, test_dataset
    
    def prepare_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        print(f"\nInitializing model: {self.model_name}")
        
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        bert_model = BertModel.from_pretrained(self.model_name)
        self.model = BERTForIntrospectiveUncertainty(bert_model, hidden_size=768)
        self.model.to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def tokenize_dataset(self, train_dataset, test_dataset):
        """Tokenize datasets"""
        print("\nTokenizing datasets...")
        
        def tokenize_function(examples):
            # Tokenize text
            tokenized = self.tokenizer(
                examples['statement'], 
                padding='max_length', 
                truncation=True, 
                max_length=128
            )
            
            # Add labels
            tokenized['truth_labels'] = examples['label']
            
            # Map uncertainty type to numeric
            # confident=0, hedged=1, explicit=2
            tokenized['uncertainty_labels'] = examples['uncertainty_type']
            
            # Add conflict labels
            tokenized['conflict_labels'] = examples['has_conflict']
            
            # Create epistemic labels (0=confident, 1=uncertain)
            tokenized['epistemic_labels'] = [
                1 if ut > 0 else 0 for ut in examples['uncertainty_type']
            ]
            
            return tokenized
        
        train_tokenized = train_dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=train_dataset.column_names
        )
        
        test_tokenized = test_dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=test_dataset.column_names
        )
        
        return train_tokenized, test_tokenized
    
    def get_training_args(self, num_epochs=5, batch_size=16):
        """Configure training arguments"""
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            report_to="none",  # Disable wandb/tensorboard
        )
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        
        # Note: predictions is a tuple of logits, labels is a tuple of label tensors
        # We'll focus on the main metrics
        
        metrics = {
            "eval_samples": len(labels[0]) if isinstance(labels, tuple) else len(labels)
        }
        
        return metrics
    
    def train(self, train_dataset, eval_dataset, num_epochs=5, batch_size=16):
        """Train the model"""
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        
        training_args = self.get_training_args(num_epochs, batch_size)
        
        # Custom trainer that handles multiple losses
        class CustomUncertaintyTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                # Extract labels
                truth_labels = inputs.pop("truth_labels")
                uncertainty_labels = inputs.pop("uncertainty_labels")
                conflict_labels = inputs.pop("conflict_labels")
                epistemic_labels = inputs.pop("epistemic_labels")
                
                # Forward pass with all labels
                loss = model(
                    **inputs,
                    truth_labels=truth_labels,
                    uncertainty_labels=uncertainty_labels,
                    conflict_labels=conflict_labels,
                    epistemic_labels=epistemic_labels
                )
                
                return (loss, None) if return_outputs else loss
        
        # Initialize trainer
        trainer = CustomUncertaintyTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        trainer.train()
        
        # Save model
        print(f"\nSaving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        
        return trainer
    
    def evaluate(self, test_dataset):
        """Detailed evaluation on test set"""
        print("\n" + "="*80)
        print("EVALUATION ON TEST SET")
        print("="*80)
        
        self.model.eval()
        
        all_truth_preds = []
        all_truth_labels = []
        all_uncertainty_preds = []
        all_uncertainty_labels = []
        all_conflict_preds = []
        all_conflict_labels = []
        
        with torch.no_grad():
            for batch in torch.utils.data.DataLoader(test_dataset, batch_size=32):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Get predictions
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                (_, sentence_logits, uncertainty_logits, conflict_logits, _, _) = outputs
                
                # Collect predictions
                truth_preds = torch.argmax(sentence_logits, dim=-1).cpu().numpy()
                uncertainty_preds = torch.argmax(uncertainty_logits, dim=-1).cpu().numpy()
                conflict_preds = torch.argmax(conflict_logits, dim=-1).cpu().numpy()
                
                all_truth_preds.extend(truth_preds)
                all_uncertainty_preds.extend(uncertainty_preds)
                all_conflict_preds.extend(conflict_preds)
                
                all_truth_labels.extend(batch['truth_labels'].numpy())
                all_uncertainty_labels.extend(batch['uncertainty_labels'].numpy())
                all_conflict_labels.extend(batch['conflict_labels'].numpy())
        
        # Print classification reports
        print("\n--- TRUTHFULNESS CLASSIFICATION ---")
        print(classification_report(
            all_truth_labels, all_truth_preds, 
            target_names=['False', 'True']
        ))
        
        print("\n--- UNCERTAINTY TYPE CLASSIFICATION ---")
        print(classification_report(
            all_uncertainty_labels, all_uncertainty_preds,
            target_names=['Confident', 'Hedged', 'Explicit']
        ))
        
        print("\n--- INTROSPECTIVE CONFLICT DETECTION ---")
        print(classification_report(
            all_conflict_labels, all_conflict_preds,
            target_names=['Consistent', 'Conflicted']
        ))
        
        # Save metrics
        metrics = {
            'truth_accuracy': np.mean(np.array(all_truth_preds) == np.array(all_truth_labels)),
            'uncertainty_accuracy': np.mean(np.array(all_uncertainty_preds) == np.array(all_uncertainty_labels)),
            'conflict_accuracy': np.mean(np.array(all_conflict_preds) == np.array(all_conflict_labels)),
        }
        
        with open(f'{self.output_dir}/eval_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics


def main():
    """Main training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train introspective uncertainty classifier')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset CSV file')
    parser.add_argument('--output_dir', type=str, default='./uncertainty_model',
                       help='Output directory for model')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='Pretrained model name')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = UncertaintyTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    # Load data
    train_dataset, test_dataset = trainer.load_data(args.data_path)
    
    # Prepare model
    trainer.prepare_model_and_tokenizer()
    
    # Tokenize
    train_tokenized, test_tokenized = trainer.tokenize_dataset(train_dataset, test_dataset)
    
    # Train
    trainer.train(train_tokenized, test_tokenized, 
                 num_epochs=args.epochs, batch_size=args.batch_size)
    
    # Evaluate
    metrics = trainer.evaluate(test_tokenized)
    
    print("\n" + "="*80)
    print("FINAL METRICS")
    print("="*80)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
