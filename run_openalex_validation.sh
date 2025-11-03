#!/bin/bash

echo "=========================================="
echo "OPENALEX NOVELTY VALIDATION"
echo "=========================================="
echo ""
echo "No API key needed!"
echo ""

# Optional: set your email for polite pool
read -p "Enter your email (optional, press Enter to skip): " user_email

if [ ! -z "$user_email" ]; then
    export OPENALEX_EMAIL="$user_email"
fi

# Run validation
python validate_novelty_openalex.py

echo ""
echo "Check results: novelty_validation_results.json"
