"""
Create Uncertainty-Augmented Dataset
Generates training data for introspective uncertainty detection
"""

import pandas as pd
import glob
import random
from pathlib import Path


class UncertaintyDatasetCreator:
    """
    Creates an augmented dataset with uncertainty communication labels
    """
    
    def __init__(self, input_dir='publicDataset copy 2'):
        self.input_dir = input_dir
        
        # Hedging templates for synthetic data generation
        self.hedge_templates = [
            # Moderate hedging
            "It is possible that {}",
            "It appears that {}",
            "Evidence suggests that {}",
            "Research indicates that {}",
            "Some sources claim that {}",
            "{}, though this is uncertain",
            "{}, but this requires verification",
            
            # Strong hedging
            "It is unclear whether {}",
            "{}, although this remains debatable",
            "{}, but further research is needed",
            "The evidence for {} is inconclusive",
            "{}, though the data is ambiguous",
            
            # Epistemic qualifiers
            "It seems that {}",
            "Apparently, {}",
            "Reportedly, {}",
            "It is believed that {}",
            "{}, according to some accounts",
        ]
    
    def load_existing_data(self):
        """Load all CSV files from input directory"""
        csv_files = glob.glob(f'{self.input_dir}/*.csv')
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.input_dir}")
        
        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                print(f"Loaded {file}: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal rows loaded: {len(combined_df)}")
        
        return combined_df
    
    def detect_existing_uncertainty(self, df):
        """
        Detect uncertainty markers in existing statements
        """
        from model.architectures.introspective_uncertainty import (
            UNCERTAINTY_MARKERS, detect_linguistic_uncertainty_markers
        )
        
        df['has_uncertainty_marker'] = 0
        df['uncertainty_type'] = 'confident'
        df['markers_found'] = ''
        
        for idx, row in df.iterrows():
            statement = str(row['statement']).lower()
            markers = detect_linguistic_uncertainty_markers(statement)
            
            if markers:
                df.at[idx, 'has_uncertainty_marker'] = 1
                df.at[idx, 'markers_found'] = str(markers)
                
                # Classify uncertainty type
                if any('explicit_uncertainty' in m for m in markers.keys()):
                    df.at[idx, 'uncertainty_type'] = 'explicit'
                elif any('research_qualifiers' in m for m in markers.keys()):
                    df.at[idx, 'uncertainty_type'] = 'explicit'
                else:
                    df.at[idx, 'uncertainty_type'] = 'hedged'
        
        print(f"\nFound {df['has_uncertainty_marker'].sum()} statements with uncertainty markers")
        return df
    
    def generate_hedged_versions(self, df, templates_per_statement=3):
        """
        Generate hedged versions of statements for training
        
        Args:
            df: Original dataframe
            templates_per_statement: Number of hedged versions per statement
        
        Returns:
            DataFrame with augmented data
        """
        new_rows = []
        
        for idx, row in df.iterrows():
            original_statement = row['statement']
            original_label = row['label']
            
            # Only create hedged versions for statements without existing hedging
            if row.get('has_uncertainty_marker', 0) == 0:
                # Sample random templates
                selected_templates = random.sample(
                    self.hedge_templates, 
                    min(templates_per_statement, len(self.hedge_templates))
                )
                
                for template in selected_templates:
                    hedged = self._apply_template(template, original_statement)
                    
                    new_rows.append({
                        'statement': hedged,
                        'label': original_label,  # Truth value unchanged
                        'has_uncertainty_marker': 1,
                        'uncertainty_type': 'hedged',
                        'is_synthetic': 1,
                        'original_statement': original_statement,
                        'template_used': template
                    })
        
        print(f"\nGenerated {len(new_rows)} hedged statements")
        return pd.DataFrame(new_rows)
    
    def _apply_template(self, template, statement):
        """Apply hedging template to statement"""
        # Remove trailing punctuation from original statement
        statement_clean = statement.rstrip('.!?')
        
        # Handle different template formats
        if '{}' in template:
            if template.startswith('{}'):
                # Template starts with statement
                hedged = template.format(statement_clean)
            else:
                # Template has statement elsewhere
                hedged = template.format(statement_clean.lower())
        else:
            # Template is a prefix
            hedged = template + " " + statement_clean.lower()
        
        # Ensure proper capitalization
        hedged = hedged[0].upper() + hedged[1:] if hedged else hedged
        
        # Add period if missing
        if not hedged.endswith(('.', '!', '?')):
            hedged += '.'
        
        return hedged
    
    def create_balanced_dataset(self, original_df, synthetic_df, balance_ratio=0.5):
        """
        Create balanced dataset with original and synthetic data
        
        Args:
            original_df: Original data with detected uncertainty
            synthetic_df: Generated hedged statements
            balance_ratio: Ratio of synthetic to original data
        
        Returns:
            Combined balanced dataset
        """
        # Add metadata to original
        original_df['is_synthetic'] = 0
        original_df['template_used'] = ''
        original_df['original_statement'] = original_df['statement']
        
        # Sample synthetic data according to balance ratio
        n_synthetic = int(len(original_df) * balance_ratio)
        synthetic_sampled = synthetic_df.sample(
            n=min(n_synthetic, len(synthetic_df)), 
            random_state=42
        )
        
        # Combine
        combined = pd.concat([original_df, synthetic_sampled], ignore_index=True)
        
        # Shuffle
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\nFinal dataset composition:")
        print(f"  Original statements: {len(original_df)}")
        print(f"  Synthetic statements: {len(synthetic_sampled)}")
        print(f"  Total: {len(combined)}")
        print(f"\nUncertainty distribution:")
        print(combined['uncertainty_type'].value_counts())
        
        return combined
    
    def add_conflict_labels(self, df):
        """
        Add introspective conflict labels
        Conflict = (true statement with hedging) OR (false statement stated confidently)
        """
        df['has_conflict'] = 0
        
        # True but hedged = author expressing genuine uncertainty
        true_hedged = (df['label'] == 1) & (df['uncertainty_type'] != 'confident')
        
        # False but confident = potential misinformation or error
        false_confident = (df['label'] == 0) & (df['uncertainty_type'] == 'confident')
        
        df.loc[true_hedged | false_confident, 'has_conflict'] = 1
        
        print(f"\nConflict statistics:")
        print(f"  Statements with conflict: {df['has_conflict'].sum()}")
        print(f"  True but hedged: {true_hedged.sum()}")
        print(f"  False but confident: {false_confident.sum()}")
        
        return df
    
    def save_dataset(self, df, output_path='enhanced_uncertainty_dataset.csv'):
        """Save the enhanced dataset"""
        df.to_csv(output_path, index=False)
        print(f"\nDataset saved to: {output_path}")
        
        # Also save train/test split
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['uncertainty_type']
        )
        
        train_path = output_path.replace('.csv', '_train.csv')
        test_path = output_path.replace('.csv', '_test.csv')
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"Train set saved to: {train_path} ({len(train_df)} rows)")
        print(f"Test set saved to: {test_path} ({len(test_df)} rows)")
        
        return train_path, test_path
    
    def create_full_dataset(self, output_path='enhanced_uncertainty_dataset.csv'):
        """
        Complete pipeline: load, detect, generate, combine, save
        """
        print("="*80)
        print("CREATING UNCERTAINTY-AUGMENTED DATASET")
        print("="*80)
        
        # Step 1: Load existing data
        print("\n[1/5] Loading existing data...")
        original_df = self.load_existing_data()
        
        # Step 2: Detect existing uncertainty markers
        print("\n[2/5] Detecting existing uncertainty markers...")
        original_df = self.detect_existing_uncertainty(original_df)
        
        # Step 3: Generate synthetic hedged versions
        print("\n[3/5] Generating synthetic hedged versions...")
        synthetic_df = self.generate_hedged_versions(original_df, templates_per_statement=3)
        
        # Step 4: Create balanced dataset
        print("\n[4/5] Creating balanced dataset...")
        combined_df = self.create_balanced_dataset(original_df, synthetic_df, balance_ratio=0.5)
        
        # Step 5: Add conflict labels
        print("\n[5/5] Adding introspective conflict labels...")
        combined_df = self.add_conflict_labels(combined_df)
        
        # Save
        train_path, test_path = self.save_dataset(combined_df, output_path)
        
        print("\n" + "="*80)
        print("DATASET CREATION COMPLETE")
        print("="*80)
        
        return combined_df, train_path, test_path


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create uncertainty-augmented dataset')
    parser.add_argument('--input_dir', type=str, default='publicDataset copy 2',
                       help='Directory containing input CSV files')
    parser.add_argument('--output', type=str, default='enhanced_uncertainty_dataset.csv',
                       help='Output CSV file path')
    parser.add_argument('--balance_ratio', type=float, default=0.5,
                       help='Ratio of synthetic to original data')
    
    args = parser.parse_args()
    
    # Create dataset
    creator = UncertaintyDatasetCreator(input_dir=args.input_dir)
    df, train_path, test_path = creator.create_full_dataset(output_path=args.output)
    
    # Print sample rows
    print("\nSample rows from final dataset:")
    print(df[['statement', 'label', 'uncertainty_type', 'has_conflict']].head(10))


if __name__ == "__main__":
    main()
