import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.core_helpers import (
    get_litellm_metadata_from_kwargs,
    get_tokens_for_tpm,
    safe_divide,
    safe_deep_copy
) 


def test_get_litellm_metadata_from_kwargs():
    kwargs = {
        "litellm_params": {
            "litellm_metadata": {},
            "metadata": {"user_api_key": "1234567890"},
        },
    }
    assert get_litellm_metadata_from_kwargs(kwargs) == {"user_api_key": "1234567890"}


def test_add_missing_spend_metadata_to_litellm_metadata():
    litellm_metadata = {"test_key": "test_value"}
    metadata = {"user_api_key_hash_value": "1234567890"}
    kwargs = {
        "litellm_params": {
            "litellm_metadata": litellm_metadata,
            "metadata": metadata,
        },
    }
    assert get_litellm_metadata_from_kwargs(kwargs) == {
        "test_key": "test_value",
        "user_api_key_hash_value": "1234567890",
    }


def test_preserve_upstream_non_openai_attributes():
    from litellm.litellm_core_utils.core_helpers import (
        preserve_upstream_non_openai_attributes,
    )
    from litellm.types.utils import ModelResponseStream

    model_response = ModelResponseStream(
        id="123",
        object="text_completion",
        created=1715811200,
        model="gpt-3.5-turbo",
    )

    setattr(model_response, "test_key", "test_value")
    preserve_upstream_non_openai_attributes(
        model_response=ModelResponseStream(),
        original_chunk=model_response,
    )

    assert model_response.test_key == "test_value"


def test_safe_divide_basic():
    """Test basic safe division functionality"""
    # Normal division
    result = safe_divide(10, 2)
    assert result == 5.0, f"Expected 5.0, got {result}"
    
    # Division with float
    result = safe_divide(7.5, 2.5)
    assert result == 3.0, f"Expected 3.0, got {result}"
    
    # Division by zero with default
    result = safe_divide(10, 0)
    assert result == 0, f"Expected 0, got {result}"
    
    # Division by zero with custom default
    result = safe_divide(10, 0, default=1)
    assert result == 1, f"Expected 1, got {result}"
    
    # Division by zero with custom default as float
    result = safe_divide(10, 0, default=0.5)
    assert result == 0.5, f"Expected 0.5, got {result}"


def test_safe_divide_edge_cases():
    """Test edge cases for safe division"""
    # Zero numerator
    result = safe_divide(0, 5)
    assert result == 0.0, f"Expected 0.0, got {result}"
    
    # Negative numbers
    result = safe_divide(-10, 2)
    assert result == -5.0, f"Expected -5.0, got {result}"
    
    # Negative denominator
    result = safe_divide(10, -2)
    assert result == -5.0, f"Expected -5.0, got {result}"
    
    # Both negative
    result = safe_divide(-10, -2)
    assert result == 5.0, f"Expected 5.0, got {result}"
    
    # Float division
    result = safe_divide(1, 3)
    assert abs(result - 0.3333333333333333) < 1e-10, f"Expected ~0.333..., got {result}"


def test_safe_divide_weight_scenario():
    """Test safe division in the context of weight calculations"""
    # Simulate weight calculation scenario
    weights = [3, 7, 0, 2]
    total_weight = sum(weights)  # 12
    
    # Normal case
    normalized_weights = [safe_divide(w, total_weight) for w in weights]
    expected = [0.25, 7/12, 0.0, 1/6]
    
    for i, (actual, exp) in enumerate(zip(normalized_weights, expected)):
        assert abs(actual - exp) < 1e-10, f"Weight {i}: Expected {exp}, got {actual}"
    
    # Zero total weight scenario (division by zero)
    zero_weights = [0, 0, 0]
    zero_total = sum(zero_weights)  # 0
    
    # Should return default values (0) for all weights
    normalized_zero_weights = [safe_divide(w, zero_total) for w in zero_weights]
    expected_zero = [0, 0, 0]
    
    assert normalized_zero_weights == expected_zero, f"Expected {expected_zero}, got {normalized_zero_weights}"


def test_safe_deep_copy_with_non_pickleables_and_span():
    """
    Verify safe_deep_copy:
    - does not crash when non-pickleables are present,
    - preserves structure/keys,
    - deep-copies JSON-y payloads (e.g., messages),
    - keeps non-pickleables by reference,
    - redacts OTEL span in the copy and restores it in the original.
    """
    import threading
    rlock = threading.RLock()
    data = {
        "metadata": {"litellm_parent_otel_span": rlock, "x": 1},
        "messages": [{"role": "user", "content": "hi"}],
        "optional_params": {"handle": rlock},
        "ok": True,
    }

    copied = safe_deep_copy(data)

    # Structure preserved
    assert set(copied.keys()) == set(data.keys())

    # Messages are deep-copied (new object, same content)
    assert copied["messages"] is not data["messages"]
    assert copied["messages"][0] == data["messages"][0]

    # Non-pickleable subtree kept by reference (no crash)
    assert copied["optional_params"] is data["optional_params"]
    assert copied["optional_params"]["handle"] is rlock

    # OTEL span: redacted in the copy, restored in original
    assert copied["metadata"]["litellm_parent_otel_span"] == "placeholder"
    assert data["metadata"]["litellm_parent_otel_span"] is rlock

    # Other simple fields unchanged
    assert copied["ok"] is True
    assert copied["metadata"]["x"] == 1


# ============================================================================
# Tests for get_tokens_for_tpm
# ============================================================================

class TestGetTokensForTpm:
    """Tests for the get_tokens_for_tpm function."""

    def test_flag_disabled_returns_total_tokens(self):
        """When exclude_cached_tokens_from_tpm is False, should return total_tokens unchanged."""
        import litellm
        original_value = litellm.exclude_cached_tokens_from_tpm
        try:
            litellm.exclude_cached_tokens_from_tpm = False
            
            # With usage object containing cached tokens
            usage = {"prompt_tokens_details": {"cached_tokens": 100}}
            result = get_tokens_for_tpm(500, usage)
            assert result == 500, f"Expected 500 when flag is disabled, got {result}"
            
            # With None usage
            result = get_tokens_for_tpm(500, None)
            assert result == 500, f"Expected 500 with None usage, got {result}"
        finally:
            litellm.exclude_cached_tokens_from_tpm = original_value

    def test_flag_enabled_with_none_usage(self):
        """When flag is enabled but usage is None, should return total_tokens."""
        import litellm
        original_value = litellm.exclude_cached_tokens_from_tpm
        try:
            litellm.exclude_cached_tokens_from_tpm = True
            
            result = get_tokens_for_tpm(500, None)
            assert result == 500, f"Expected 500 with None usage, got {result}"
        finally:
            litellm.exclude_cached_tokens_from_tpm = original_value

    def test_dict_usage_with_prompt_tokens_details(self):
        """Test with dict usage containing prompt_tokens_details (Chat Completions API)."""
        import litellm
        original_value = litellm.exclude_cached_tokens_from_tpm
        try:
            litellm.exclude_cached_tokens_from_tpm = True
            
            usage = {
                "prompt_tokens": 400,
                "completion_tokens": 100,
                "total_tokens": 500,
                "prompt_tokens_details": {
                    "cached_tokens": 200
                }
            }
            result = get_tokens_for_tpm(500, usage)
            assert result == 300, f"Expected 500-200=300, got {result}"
        finally:
            litellm.exclude_cached_tokens_from_tpm = original_value

    def test_dict_usage_with_input_tokens_details(self):
        """Test with dict usage containing input_tokens_details (Responses API)."""
        import litellm
        original_value = litellm.exclude_cached_tokens_from_tpm
        try:
            litellm.exclude_cached_tokens_from_tpm = True
            
            usage = {
                "input_tokens": 400,
                "output_tokens": 100,
                "total_tokens": 500,
                "input_tokens_details": {
                    "cached_tokens": 150
                }
            }
            result = get_tokens_for_tpm(500, usage)
            assert result == 350, f"Expected 500-150=350, got {result}"
        finally:
            litellm.exclude_cached_tokens_from_tpm = original_value

    def test_dict_usage_input_tokens_details_takes_priority(self):
        """Test that input_tokens_details is checked before prompt_tokens_details."""
        import litellm
        original_value = litellm.exclude_cached_tokens_from_tpm
        try:
            litellm.exclude_cached_tokens_from_tpm = True
            
            # Both present - input_tokens_details should be used
            usage = {
                "total_tokens": 500,
                "input_tokens_details": {
                    "cached_tokens": 100
                },
                "prompt_tokens_details": {
                    "cached_tokens": 200
                }
            }
            result = get_tokens_for_tpm(500, usage)
            assert result == 400, f"Expected 500-100=400 (input_tokens_details), got {result}"
        finally:
            litellm.exclude_cached_tokens_from_tpm = original_value

    def test_usage_object_with_prompt_tokens_details(self):
        """Test with Usage object containing prompt_tokens_details."""
        import litellm
        from litellm.types.utils import Usage, PromptTokensDetailsWrapper
        
        original_value = litellm.exclude_cached_tokens_from_tpm
        try:
            litellm.exclude_cached_tokens_from_tpm = True
            
            usage = Usage(
                prompt_tokens=400,
                completion_tokens=100,
                total_tokens=500,
                prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=250)
            )
            result = get_tokens_for_tpm(500, usage)
            assert result == 250, f"Expected 500-250=250, got {result}"
        finally:
            litellm.exclude_cached_tokens_from_tpm = original_value

    def test_no_cached_tokens_returns_total(self):
        """Test when there are no cached tokens in usage."""
        import litellm
        original_value = litellm.exclude_cached_tokens_from_tpm
        try:
            litellm.exclude_cached_tokens_from_tpm = True
            
            # Dict without cached_tokens
            usage = {
                "prompt_tokens": 400,
                "completion_tokens": 100,
                "total_tokens": 500,
                "prompt_tokens_details": {}
            }
            result = get_tokens_for_tpm(500, usage)
            assert result == 500, f"Expected 500 (no cached tokens), got {result}"
            
            # Dict with cached_tokens = 0
            usage = {
                "total_tokens": 500,
                "prompt_tokens_details": {"cached_tokens": 0}
            }
            result = get_tokens_for_tpm(500, usage)
            assert result == 500, f"Expected 500 (cached_tokens=0), got {result}"
        finally:
            litellm.exclude_cached_tokens_from_tpm = original_value

    def test_cached_tokens_greater_than_total_returns_zero(self):
        """Test that result is never negative (uses max(0, ...))."""
        import litellm
        original_value = litellm.exclude_cached_tokens_from_tpm
        try:
            litellm.exclude_cached_tokens_from_tpm = True
            
            # Edge case: cached_tokens > total_tokens (shouldn't happen, but handle gracefully)
            usage = {
                "total_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 200}
            }
            result = get_tokens_for_tpm(100, usage)
            assert result == 0, f"Expected 0 (not negative), got {result}"
        finally:
            litellm.exclude_cached_tokens_from_tpm = original_value

    def test_cached_tokens_none_treated_as_zero(self):
        """Test that None cached_tokens is treated as 0."""
        import litellm
        original_value = litellm.exclude_cached_tokens_from_tpm
        try:
            litellm.exclude_cached_tokens_from_tpm = True
            
            usage = {
                "total_tokens": 500,
                "prompt_tokens_details": {"cached_tokens": None}
            }
            result = get_tokens_for_tpm(500, usage)
            assert result == 500, f"Expected 500 (None treated as 0), got {result}"
        finally:
            litellm.exclude_cached_tokens_from_tpm = original_value

    def test_fallback_to_prompt_tokens_details_when_input_is_zero(self):
        """Test fallback to prompt_tokens_details when input_tokens_details has 0 cached."""
        import litellm
        original_value = litellm.exclude_cached_tokens_from_tpm
        try:
            litellm.exclude_cached_tokens_from_tpm = True
            
            usage = {
                "total_tokens": 500,
                "input_tokens_details": {"cached_tokens": 0},
                "prompt_tokens_details": {"cached_tokens": 150}
            }
            result = get_tokens_for_tpm(500, usage)
            assert result == 350, f"Expected 500-150=350 (fallback to prompt_tokens_details), got {result}"
        finally:
            litellm.exclude_cached_tokens_from_tpm = original_value
