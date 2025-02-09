import pytest
from unittest.mock import patch, MagicMock

import sys
sys.path.append(".")  # Add the src folder to the Python path

from src.generate import load_llama, simple_call_llama  # Adjust this path to your module

simple_prompt = "Say: Hello"

def test_load_llama():
    """Test loading the LLAMA model with a mocked pipeline."""
    # Load the model (which now returns the mocked pipeline)
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    model = load_llama(model_id)
    
    # Check that the model is a pipeline
    assert model is not None

    assert simple_call_llama(model, simple_prompt, temperature=1e-5) == "Hello"
    
