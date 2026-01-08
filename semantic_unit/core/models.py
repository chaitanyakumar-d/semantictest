"""
Pydantic models for semantic evaluation results.

This module defines the data structures used to represent evaluation outcomes,
including drift detection metrics and semantic alignment scores.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class DriftResult(BaseModel):
    """
    Represents the semantic drift measurement between actual and expected outputs.

    This model quantifies the divergence in semantic meaning through distributional
    alignment scores and provides interpretable reasoning for the measured drift.

    Attributes
    ----------
    score : float
        Semantic alignment coefficient ranging from 0.0 (complete divergence) to
        1.0 (perfect alignment). This represents the cosine similarity in the
        latent semantic space between the actual and expected representations.
    reasoning : str
        Detailed explanation of the semantic entropy analysis, including specific
        factual discrepancies, logical inconsistencies, or contextual drift patterns
        identified during evaluation.
    actual : str
        The observed output text being evaluated for semantic drift.
    expected : str
        The reference ground truth text representing the ideal semantic content.
    model : str, optional
        The language model identifier used for semantic evaluation (e.g., 'gpt-4o-mini').
    metadata : dict, optional
        Additional evaluation metadata including token counts, latency metrics,
        and confidence intervals.

    Examples
    --------
    >>> result = DriftResult(
    ...     score=0.95,
    ...     reasoning="High semantic alignment with minor lexical variation",
    ...     actual="The experiment succeeded",
    ...     expected="The experiment was successful"
    ... )
    """

    score: float = Field(
        ..., ge=0.0, le=1.0, description="Semantic alignment score between 0.0 and 1.0"
    )
    reasoning: str = Field(
        ..., min_length=1, description="Detailed explanation of the semantic evaluation"
    )
    actual: str = Field(..., description="The actual output text being evaluated")
    expected: str = Field(..., description="The expected reference text")
    model: Optional[str] = Field(default=None, description="Language model used for evaluation")
    metadata: Optional[dict] = Field(default=None, description="Additional evaluation metadata")

    @field_validator("score")
    @classmethod
    def validate_score_bounds(cls, v: float) -> float:
        """
        Validate that the semantic alignment score falls within valid bounds.

        Parameters
        ----------
        v : float
            The score value to validate

        Returns
        -------
        float
            The validated score

        Raises
        ------
        ValueError
            If score is outside the [0.0, 1.0] interval
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {v}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "score": 0.92,
                "reasoning": "Both texts convey identical factual information with minor syntactic variation. Semantic entropy is minimal.",
                "actual": "The model achieved 95% accuracy",
                "expected": "Model accuracy reached 95%",
                "model": "gpt-4o-mini",
                "metadata": {"tokens": 45, "latency_ms": 234},
            }
        }
    }
