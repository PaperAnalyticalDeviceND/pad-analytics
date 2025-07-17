# README Updates Needed

## Summary
Most README examples work correctly with the enhanced API functions. Only two minor updates needed:

## Required Updates

### 1. Line 56: Quick Start Example
**Current:**
```python
cards = pad.get_project_cards(project_name="ChemoPADNNtraining2024")
```

**Should be:**
```python
cards = pad.get_project_cards("ChemoPADNNtraining2024")
```

### 2. Line 78: Data Exploration Example  
**Current:**
```python
cards = pad.get_card_by_sample_id(65490) # sample id
```

**Should be:**
```python
card = pad.get_card(sample_id=65490) # sample id
```

### 3. Line 107: Visualization Example
**Current:**
```python
cards_df = pad.get_project_cards("ChemoPADNNtraining2024")
```

**Already correct syntax** - no change needed

## Test Results
✅ All functions work correctly with enhanced API
✅ Error handling works properly (shows user-friendly messages)
✅ Visualization functions execute successfully
✅ Both NN and PLS model predictions work correctly
✅ All parameter variations are supported

## Note
The deprecated function `get_card_by_sample_id()` is now internal (`_get_card_by_sample_id()`) and users should use `get_card(sample_id=...)` instead.