# LLM Self-Detection Research Template

This template enables the AI Scientist to use the seed_ideas.json you generated with the custom model and generate new research according to its own pipeline, including paper review.
There is some setup involved such as preparing a starter experiment and plotting files. You need a semantic Scholar api key to detect novelty scores. 
## Quick Start

### 1. Clone and Setup AI Scientist

```bash
# Clone the repository
git clone https://github.com/SakanaAI/AI-Scientist.git
cd AI-Scientist

# Create and activate virtual environment (recommended)
python -m venv ai_scientist_env
source ai_scientist_env/bin/activate  # Linux/Mac
# ai_scientist_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install LaTeX (required for paper generation)
# Option 1: Minimal LaTeX (~500MB)
apt-get update && apt-get install -y texlive-latex-base texlive-latex-extra

# If you don't have a semantic scholar api use this
export OPENALEX_MAIL_ADDRESS="your-email@example.com"
# Set your API keys
export ANTHROPIC_API_KEY="your-claude-key"   # For Claude models
export S2_API_KEY="your-semantic-scholar-key" # Optional but recommended
```

### 2. Install the LLM Self-Detection Template

```bash
# Create template directory
mkdir -p templates/llm_self_detection

# Navigate to template directory
cd templates/llm_self_detection

# Create required files:
# 1. seed_ideas.json - Copy the 8 seed ideas JSON generated from the custom model
# 2. prompt.json - Task description and system prompt
# 3. experiment.py - Base experimental code (AI Scientist will modify this)
# 4. plot.py - Base plotting code (AI Scientist will modify this)
```

### 3. Required Template Files

**seed_ideas.json** - Contains 8 research directions for LLM falsehood detection

**prompt.json**:
```json
{
    "system": "You are an ambitious AI researcher focused on understanding and improving the reliability of large language models. Your goal is to develop novel methods for detecting when LLMs generate inaccurate or false information by analyzing their internal mechanisms.",
    "task_description": "You are investigating how large language models can detect their own generation of inaccurate information. The research focuses on analyzing internal states, activation patterns, and confidence levels to identify when models are generating false statements."
}
```

**experiment.py** (minimal base):
```python
import argparse
import json
import os
import numpy as np

def run_experiment():
    # AI Scientist will modify this
    results = {"accuracy": np.random.rand()}
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    
    results = run_experiment()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
```

**plot.py** (minimal base):
```python
import matplotlib.pyplot as plt
import os

def create_plots(results_dir, plot_dir):
    # AI Scientist will modify this
    plt.figure()
    plt.text(0.5, 0.5, 'Placeholder')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'placeholder.png'))
    plt.close()

if __name__ == "__main__":
    create_plots("results", "plots")
```

### 4. Final Template Structure

```
templates/llm_self_detection/
├── seed_ideas.json    # 8 seed ideas for LLM self-detection research
├── prompt.json        # System prompt and task description
├── experiment.py      # Base experimental framework
└── plot.py            # Results visualization
```

### 5. Run the AI Scientist

```bash
# Return to main directory
cd /workspace/AI-Scientist

# Generate and test 1 idea (small test run)
python launch_scientist.py --model "claude-3-5-sonnet-20241022" --experiment llm_self_detection --num-ideas 1

# Skip novelty checking (if no Semantic Scholar API)
python launch_scientist.py --model "claude-3-5-sonnet-20241022" --experiment llm_self_detection --num-ideas 1 --skip-novelty-check

# Run without paper writing (faster testing)
python launch_scientist.py --model "claude-3-5-sonnet-20241022" --experiment llm_self_detection --num-ideas 1 --no-writeup

# Generate multiple ideas
python launch_scientist.py --model "claude-3-5-sonnet-20241022" --experiment llm_self_detection --num-ideas 5
```

## Expected Outputs

After running, check the `results/llm_self_detection/` directory for:
- Generated experiment code
- Results and metrics
- Research papers (if writeup enabled)
- Visualizations and plots

## Troubleshooting

### Common Issues

**ModuleNotFoundError**: Make sure you're in the virtual environment and have installed all requirements:
```bash
# Activate virtual environment
source ai_scientist_env/bin/activate  # Linux/Mac

# Reinstall requirements
pip install -r requirements.txt
```

**API Key Errors**: Ensure your API key is set correctly:
```bash
# For Claude
export ANTHROPIC_API_KEY="sk-ant-..."


**LaTeX Errors**: The AI Scientist requires LaTeX for paper generation. To skip this:
```bash
python launch_scientist.py --model "claude-3-5-sonnet-20241022" --experiment llm_self_detection --num-ideas 1 --no-writeup
```
