"""
Test suite for data models.

This module contains unit tests for Pydantic models used in semantic evaluation.
"""

import pytest
from pydantic import ValidationError
from semantic_unit.core.models import DriftResult


class TestDriftResult:
    """Test cases for the DriftResult model."""

    def test_valid_drift_result(self):
        """Test creation of a valid DriftResult."""
        result = DriftResult(
            score=0.95,
            reasoning="High semantic alignment",
            actual="Test output",
            expected="Expected output",
        )
        assert result.score == 0.95
        assert result.reasoning == "High semantic alignment"
        assert result.actual == "Test output"
        assert result.expected == "Expected output"

    def test_drift_result_with_model(self):
        """Test DriftResult with model field."""
        result = DriftResult(
            score=0.88,
            reasoning="Good alignment",
            actual="Output A",
            expected="Output B",
            model="gpt-4o-mini",
        )
        assert result.model == "gpt-4o-mini"

    def test_drift_result_with_metadata(self):
        """Test DriftResult with metadata."""
        metadata = {"tokens": 150, "latency_ms": 230}
        result = DriftResult(
            score=0.75,
            reasoning="Moderate alignment",
            actual="Text A",
            expected="Text B",
            metadata=metadata,
        )
        assert result.metadata == metadata
        assert result.metadata["tokens"] == 150

    def test_score_bounds_validation_high(self):
        """Test score validation for values above 1.0."""
        with pytest.raises(ValidationError):
            DriftResult(score=1.5, reasoning="Invalid score", actual="A", expected="B")

    def test_score_bounds_validation_low(self):
        """Test score validation for negative values."""
        with pytest.raises(ValidationError):
            DriftResult(score=-0.1, reasoning="Invalid score", actual="A", expected="B")

    def test_score_exact_bounds(self):
        """Test score validation at exact boundaries."""
        # Test lower bound
        result_low = DriftResult(score=0.0, reasoning="Zero alignment", actual="A", expected="B")
        assert result_low.score == 0.0

        # Test upper bound
        result_high = DriftResult(
            score=1.0, reasoning="Perfect alignment", actual="A", expected="B"
        )
        assert result_high.score == 1.0

    def test_missing_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            DriftResult(score=0.5, reasoning="Test")  # Missing actual and expected

    def test_empty_reasoning(self):
        """Test validation of empty reasoning string."""
        with pytest.raises(ValidationError):
            DriftResult(score=0.5, reasoning="", actual="A", expected="B")

    def test_model_serialization(self):
        """Test JSON serialization of DriftResult."""
        result = DriftResult(
            score=0.92,
            reasoning="High alignment",
            actual="Output",
            expected="Expected",
            model="gpt-4o-mini",
            metadata={"test": "value"},
        )

        json_data = result.model_dump()
        assert json_data["score"] == 0.92
        assert json_data["reasoning"] == "High alignment"
        assert json_data["model"] == "gpt-4o-mini"
        assert json_data["metadata"]["test"] == "value"

    def test_model_json_export(self):
        """Test JSON string export."""
        result = DriftResult(score=0.85, reasoning="Test reasoning", actual="A", expected="B")

        json_str = result.model_dump_json()
        assert isinstance(json_str, str)
        assert "0.85" in json_str
        assert "Test reasoning" in json_str
