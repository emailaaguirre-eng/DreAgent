# Model Assignments Reference

## Current Model Code Location

The model assignments are defined in `Lea_Visual_Code_v2.5.1a_ TTS.py` at **line 1393**:

```python
MODE_MODEL_DEFAULTS = {
    "General Assistant & Triage": ("gpt-5-mini", "gpt-5.1"),
    "IT Support": ("gpt-5.1", "gpt-5-mini"),
    "Executive Assistant & Operations": ("gpt-5-mini", "gpt-5.1"),
    "Incentives": ("gpt-5.1", "gpt-5-mini"),
    "Research & Learning": ("gpt-5.1", "gpt-5-mini"),
    "Legal Research Assistant": ("gpt-5.1", "gpt-5"),
    "Accounting/Finance/Taxes": ("gpt-5.1", "gpt-5-mini"),
}
```

## Model Assignment Format

Each entry follows this format:
```python
"Mode Name": ("primary_model", "backup_model")
```

- **First model** = Default/primary model for that mode
- **Second model** = Backup model if primary is unavailable

## Your Requested Assignments

| Mode | Primary Model | Backup Model |
|------|--------------|--------------|
| General Assistant & Triage | gpt-5-mini | gpt-5.1 |
| IT Support | gpt-5.1 | gpt-5-mini |
| Executive Assistant & Operations | gpt-5-mini | gpt-5.1 |
| Incentives | gpt-5.1 | gpt-5-mini |
| Research & Learning | gpt-5.1 | gpt-5-mini |
| Legal Research Assistant | gpt-5.1 | gpt-5 |
| Accounting/Finance/Taxes | gpt-5.1 | gpt-5-mini |

## How It Works

1. **When a mode is selected**, `on_mode_changed()` is called
2. **It calls** `get_default_model_for_mode(mode)` which:
   - Looks up the mode in `MODE_MODEL_DEFAULTS`
   - Returns the primary model (first in tuple)
   - Falls back to backup model if primary isn't available
3. **The model dropdown** is updated to show the assigned model
4. **If the model isn't in the dropdown**, it's added automatically from `MODE_MODEL_DEFAULTS`

## Why Models Might Not Show

If models don't appear in the dropdown, it could be because:

1. **Model registry unavailable** - The code now ensures models from `MODE_MODEL_DEFAULTS` are always added
2. **Model names don't match** - Check that model IDs match exactly (e.g., "gpt-5-mini" not "gpt-5mini")
3. **Registry filtering** - The model registry might filter out unavailable models

## Fix Applied

The code now:
- ✅ Defines `MODE_MODEL_DEFAULTS` BEFORE building model options
- ✅ Automatically adds models from `MODE_MODEL_DEFAULTS` to the dropdown
- ✅ Ensures requested models are always available, even if not in registry

## Verifying It Works

1. **Check the code** - Line 1393 should have `MODE_MODEL_DEFAULTS` with your models
2. **Run the program** - Switch modes and verify correct models are selected
3. **Check dropdown** - Models like "GPT-5 Mini" and "GPT-5.1" should appear
4. **Check logs** - Look for messages like "Added requested model to options"

## Changing Model Assignments

To change model assignments, edit `MODE_MODEL_DEFAULTS` at line 1393:

```python
MODE_MODEL_DEFAULTS = {
    "Mode Name": ("new_primary_model", "new_backup_model"),
    # ... other modes
}
```

The changes will take effect immediately - no restart needed for the mapping, but you may need to refresh the model list.

