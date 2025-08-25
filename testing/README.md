# Truthfulness Analysis Flask API

A Flask API that analyzes academic papers to identify novel/uncertain sentences using a fine-tuned BERT model.

## Installation

```bash
pip install flask flask-cors numpy torch transformers safetensors requests PyPDF2 scipy scikit-learn python-dotenv waitress anthropic
```

## Environment Setup

Create a `.env` file:

```env
ANTHROPIC_API_KEY=your_api_key_here
MODEL_PATH=./saved_model
PORT=8888
```

## Running the API

```bash
python app.py
```

## Testing the Endpoint

```bash
curl -X POST http://localhost:8888/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://arxiv.org/abs/2304.13734"}' | jq '.'
```
