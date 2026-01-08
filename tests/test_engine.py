"""
Test suite for the semantic evaluation engine.

This module contains unit tests for the SemanticJudge class and related
evaluation functionality.
"""

from unittest.mock import Mock, patch

import pytest

from semantic_unit.core.engine import SemanticJudge
from semantic_unit.core.models import DriftResult


class TestSemanticJudge:
    """Test cases for the SemanticJudge evaluation engine."""

    def test_initialization_default(self):
        """Test SemanticJudge initialization with default parameters."""
        judge = SemanticJudge()
        assert judge.model == "gpt-4o-mini"
        assert judge.temperature == 0.0
        assert judge.max_tokens == 500
        assert judge.system_prompt is not None

    def test_initialization_custom(self):
        """Test SemanticJudge initialization with custom parameters."""
        judge = SemanticJudge(model="gpt-4", temperature=0.3, max_tokens=1000)
        assert judge.model == "gpt-4"
        assert judge.temperature == 0.3
        assert judge.max_tokens == 1000

    def test_evaluate_validates_empty_actual(self):
        """Test that evaluate raises ValueError for empty actual text."""
        judge = SemanticJudge()
        with pytest.raises(ValueError, match="Actual text cannot be empty"):
            judge.evaluate("", "expected text")

    def test_evaluate_validates_empty_expected(self):
        """Test that evaluate raises ValueError for empty expected text."""
        judge = SemanticJudge()
        with pytest.raises(ValueError, match="Expected text cannot be empty"):
            judge.evaluate("actual text", "")

    @patch("semantic_unit.core.engine.litellm.completion")
    def test_evaluate_success(self, mock_completion):
        """Test successful semantic evaluation."""
        # Mock LiteLLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            '{"score": 0.95, "reasoning": "High semantic alignment observed"}'
        )
        mock_response.usage.total_tokens = 150
        mock_completion.return_value = mock_response

        judge = SemanticJudge()
        result = judge.evaluate(
            actual="The test passed successfully", expected="The test was successful"
        )

        assert isinstance(result, DriftResult)
        assert result.score == 0.95
        assert "semantic alignment" in result.reasoning.lower()
        assert result.actual == "The test passed successfully"
        assert result.expected == "The test was successful"
        assert result.model == "gpt-4o-mini"

    @patch("semantic_unit.core.engine.litellm.completion")
    def test_evaluate_with_metadata(self, mock_completion):
        """Test evaluation with custom metadata."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            '{"score": 0.88, "reasoning": "Minor drift detected"}'
        )
        mock_response.usage.total_tokens = 120
        mock_completion.return_value = mock_response

        judge = SemanticJudge()
        custom_metadata = {"task": "classification", "domain": "medical"}
        result = judge.evaluate(
            actual="Patient has fever",
            expected="Patient exhibits pyrexia",
            metadata=custom_metadata,
        )

        assert result.metadata is not None
        assert result.metadata["task"] == "classification"
        assert result.metadata["domain"] == "medical"

    @patch("semantic_unit.core.engine.litellm.completion")
    def test_evaluate_invalid_json(self, mock_completion):
        """Test handling of invalid JSON response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Not a valid JSON"
        mock_completion.return_value = mock_response

        judge = SemanticJudge()
        with pytest.raises(RuntimeError, match="Failed to parse LLM response"):
            judge.evaluate("actual", "expected")

    @patch("semantic_unit.core.engine.litellm.completion")
    def test_evaluate_missing_fields(self, mock_completion):
        """Test handling of response missing required fields."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"score": 0.5}'  # Missing reasoning
        mock_completion.return_value = mock_response

        judge = SemanticJudge()
        with pytest.raises(RuntimeError, match="Semantic evaluation failed"):
            judge.evaluate("actual", "expected")

    @patch("semantic_unit.core.engine.litellm.completion")
    def test_batch_evaluate(self, mock_completion):
        """Test batch evaluation of multiple pairs."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"score": 0.9, "reasoning": "Test reasoning"}'
        mock_response.usage.total_tokens = 100
        mock_completion.return_value = mock_response

        judge = SemanticJudge()
        pairs = [("actual1", "expected1"), ("actual2", "expected2"), ("actual3", "expected3")]

        results = judge.batch_evaluate(pairs)

        assert len(results) == 3
        assert all(isinstance(r, DriftResult) for r in results)
        assert mock_completion.call_count == 3

    def test_construct_prompt(self):
        """Test prompt construction."""
        judge = SemanticJudge()
        prompt = judge._construct_prompt("actual text", "expected text")

        assert "actual text" in prompt
        assert "expected text" in prompt
        assert "EXPECTED" in prompt
        assert "ACTUAL" in prompt
        assert "semantic" in prompt.lower()
