# Truthfulness Detector

Analyzes scientific papers to find uncertain claims and generate research ideas.

## Folders

**`/model`** - Build the truthfulness detection model  
**`/testing`** - Test the model with Flask API  
**`/sakana_integration`** - Connect to SAKANA AI Scientist  

## Workflow

1. Train model with `model/construct_truthfulness_model.py`
2. Test with `testing/app.py` 
3. Generate seed ideas for SAKANA using instructions in `sakana_integration/HOWTO.md`

See individual README files in each folder for details.
