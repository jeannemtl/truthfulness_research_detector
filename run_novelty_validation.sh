#!/bin/bash

echo "=========================================="
echo "SEMANTIC SCHOLAR NOVELTY VALIDATION"
echo "=========================================="
echo ""

# Check if seed_ideas.json exists
if [ ! -f "seed_ideas.json" ]; then
    echo "Error: seed_ideas.json not found!"
    echo "Please run this from /workspace/truthfulness_research_detector/"
    exit 1
fi

# Check for API key
if [ -z "$S2_API_KEY" ]; then
    echo "No Semantic Scholar API key found."
    echo ""
    echo "Options:"
    echo "1. Continue with public API (rate limited, slower)"
    echo "2. Get free API key at https://www.semanticscholar.org/product/api"
    echo "   Then: export S2_API_KEY='your-key'"
    echo ""
    read -p "Continue with public API? [y/N]: " continue
    if [[ ! $continue =~ ^[Yy]$ ]]; then
        exit 0
    fi
    echo ""
fi

# Run validation
python validate_novelty_semantic_scholar.py

echo ""
echo "Validation complete!"
echo "Check: novelty_validation_results.json"
