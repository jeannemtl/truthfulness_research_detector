# Generating Seed Ideas with Introspective Relevance

## Usage

### 1. Generate with Introspective Relevance (DEFAULT)
```bash
python generate_seed_ideas_introspective.py --introspective
# Output: seed_ideas_introspective.json
```

### 2. Generate with Baseline (Cosine Similarity)
```bash
python generate_seed_ideas_introspective.py --baseline
# Output: seed_ideas_baseline.json
```

### 3. Generate with BOTH and Compare
```bash
python generate_seed_ideas_introspective.py --both
# Output: seed_ideas_introspective.json + seed_ideas_baseline.json
# Shows comparison at end
```

## What's Different?

**Introspective:**
- Uses epistemic certainty head
- Model judges: "Am I certain about the relationship?"
- TRUE Lindsey-style introspection

**Baseline:**
- Uses cosine similarity
- Mathematical calculation
- NOT introspective

## Expected Output
```
================================================================================
COMPLETE!
================================================================================

INTROSPECTIVE:
  Output: seed_ideas_introspective.json
  Ideas: 8
  Mean combined score: 0.XXX
  Mean relevance: 0.XXX

BASELINE:
  Output: seed_ideas_baseline.json
  Ideas: 8
  Mean combined score: 0.XXX
  Mean relevance: 0.XXX

================================================================================
COMPARISON
================================================================================

Mean combined scores:
  Introspective: 0.XXX
  Baseline:      0.XXX

✓ Introspective is +X.X% better!
```

## Key Changes

1. **`get_context_relevance_introspective()`** - NEW!
   - Uses epistemic certainty head
   - True introspection

2. **`analyze_paper_introspective()`** - UPDATED!
   - Accepts `use_introspective` flag
   - Supports both methods

3. **Command-line arguments** - NEW!
   - `--introspective`: Use epistemic certainty
   - `--baseline`: Use cosine similarity
   - `--both`: Generate and compare both

