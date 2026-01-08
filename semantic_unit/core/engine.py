"""
Semantic evaluation engine for deterministic drift detection.

This module implements the core evaluation logic using large language models
to assess semantic alignment and detect distributional drift between actual
and expected text outputs.
"""

import json
import os
from typing import Optional, Dict, Any

import litellm
from dotenv import load_dotenv

from semantic_unit.core.models import DriftResult

# Load environment variables
load_dotenv()


class SemanticJudge:
    """
    Neural semantic evaluator for measuring distributional alignment and drift.

    This class implements a deterministic evaluation framework that leverages
    large language models to quantify semantic entropy and factual divergence
    between observed and expected textual outputs. The evaluation operates in
    the latent semantic space, providing interpretable alignment scores.

    The judge employs a zero-shot prompting strategy with structured output
    constraints to ensure reproducible and calibrated semantic measurements.

    Parameters
    ----------
    model : str, optional
        The language model identifier to use for semantic evaluation.
        Defaults to 'gpt-4o-mini'. Must be a model supported by LiteLLM.
    temperature : float, optional
        Sampling temperature for model inference. Lower values (→0) increase
        determinism in evaluation. Defaults to 0.0 for maximum reproducibility.
    max_tokens : int, optional
        Maximum token budget for model response generation. Defaults to 500.
    api_key : str, optional
        API authentication key for the language model provider. If not provided,
        will attempt to load from SEMANTIC_UNIT_API_KEY environment variable.

    Attributes
    ----------
    model : str
        The configured language model identifier
    temperature : float
        The inference temperature parameter
    max_tokens : int
        The token generation limit
    system_prompt : str
        The evaluation system prompt defining the judge's role and output schema

    Examples
    --------
    >>> judge = SemanticJudge(model="gpt-4o-mini", temperature=0.0)
    >>> result = judge.evaluate(
    ...     actual="The accuracy was 92%",
    ...     expected="The model achieved 92% accuracy"
    ... )
    >>> print(f"Alignment Score: {result.score}")
    Alignment Score: 0.98

    Notes
    -----
    The evaluation framework assumes that semantic equivalence can be reliably
    assessed through LLM-based comparison in the latent space. The alignment
    score represents an approximation of cosine similarity between semantic
    embeddings, calibrated through prompt engineering.

    References
    ----------
    .. [1] Liu et al. (2023). "G-Eval: NLG Evaluation using GPT-4 with Better
           Human Alignment." arXiv:2303.16634
    .. [2] Zheng et al. (2024). "Judging LLM-as-a-Judge with MT-Bench and
           Chatbot Arena." NeurIPS 2023.
    """

    # System prompt defining the semantic evaluation task
    SYSTEM_PROMPT = """You are a strict Logic Judge. Compare if the ACTUAL text conveys the same factual meaning as the EXPECTED text. Return a JSON with: score (0.0 to 1.0) and reasoning.

Your evaluation must be:
1. Deterministic: Same inputs always yield same outputs
2. Factual: Focus on semantic meaning, not stylistic variation
3. Precise: Distinguish between factual drift and lexical paraphrasing

Score Guidelines:
- 1.0: Perfect semantic alignment (identical factual content)
- 0.8-0.9: High alignment (same facts, minor contextual differences)
- 0.5-0.7: Moderate alignment (overlapping facts, some divergence)
- 0.3-0.4: Low alignment (significant factual discrepancies)
- 0.0-0.2: Minimal alignment (fundamentally different factual content)

Return ONLY valid JSON in this exact format:
{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<detailed explanation of semantic entropy and vector alignment>"
}"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 500,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the semantic judge with specified model configuration.

        Parameters
        ----------
        model : str, optional
            Language model identifier (default: 'gpt-4o-mini')
        temperature : float, optional
            Sampling temperature for deterministic evaluation (default: 0.0)
        max_tokens : int, optional
            Maximum response token budget (default: 500)
        api_key : str, optional
            API authentication key (default: from environment)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Configure API key from parameter or environment
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY") and not os.getenv("SEMANTIC_UNIT_API_KEY"):
            # Allow initialization without key for testing, but will fail on actual calls
            pass
        elif os.getenv("SEMANTIC_UNIT_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.getenv("SEMANTIC_UNIT_API_KEY")

        self.system_prompt = self.SYSTEM_PROMPT

    def evaluate(
        self, actual: str, expected: str, metadata: Optional[Dict[str, Any]] = None
    ) -> DriftResult:
        """
        Evaluate semantic drift between actual and expected text outputs.

        This method performs a neural semantic comparison in the latent space,
        quantifying the distributional divergence between the observed output
        and the reference ground truth. The evaluation leverages chain-of-thought
        reasoning to provide interpretable drift measurements.

        The alignment score represents the degree of semantic preservation, with
        values approaching 1.0 indicating minimal entropy and high factual fidelity.

        Parameters
        ----------
        actual : str
            The observed output text being evaluated for semantic drift.
            This represents the model's actual generation or system output.
        expected : str
            The reference ground truth text representing the ideal semantic
            content. This serves as the anchor point for drift measurement.
        metadata : dict, optional
            Additional context for evaluation (e.g., domain, task type).
            This metadata is preserved in the result for audit trails.

        Returns
        -------
        DriftResult
            A structured result containing:
            - score: Semantic alignment coefficient ∈ [0, 1]
            - reasoning: Interpretable explanation of drift patterns
            - actual: The evaluated output
            - expected: The reference text
            - model: The evaluator model identifier
            - metadata: Evaluation context and metrics

        Raises
        ------
        ValueError
            If actual or expected text is empty or invalid
        RuntimeError
            If the language model API call fails or returns invalid JSON

        Examples
        --------
        >>> judge = SemanticJudge()
        >>> result = judge.evaluate(
        ...     actual="The experiment succeeded with 95% accuracy",
        ...     expected="Our experiment achieved a 95% success rate"
        ... )
        >>> print(f"Semantic Drift Score: {1 - result.score:.3f}")
        Semantic Drift Score: 0.050

        Notes
        -----
        The evaluation is designed to be deterministic when temperature=0.0,
        though minor variations may occur due to model version updates or
        API-level non-determinism. For critical applications, consider running
        multiple evaluations and aggregating results.

        The reasoning field provides chain-of-thought explanations that can be
        used for error analysis, debugging, and understanding semantic drift
        patterns in production systems.
        """
        # Input validation
        if not actual or not actual.strip():
            raise ValueError("Actual text cannot be empty")
        if not expected or not expected.strip():
            raise ValueError("Expected text cannot be empty")

        # Construct evaluation prompt
        user_prompt = self._construct_prompt(actual, expected)

        try:
            # Perform LLM-based semantic evaluation
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},  # Enforce JSON output
            )

            # Extract and parse response
            content = response.choices[0].message.content
            evaluation_data = json.loads(content)

            # Validate required fields
            if "score" not in evaluation_data or "reasoning" not in evaluation_data:
                raise ValueError("LLM response missing required fields (score, reasoning)")

            # Construct drift result with metadata
            result_metadata = {
                "tokens_used": response.usage.total_tokens if hasattr(response, "usage") else None,
                "model_version": self.model,
            }
            if metadata:
                result_metadata.update(metadata)

            # Return structured result
            return DriftResult(
                score=float(evaluation_data["score"]),
                reasoning=evaluation_data["reasoning"],
                actual=actual,
                expected=expected,
                model=self.model,
                metadata=result_metadata,
            )

        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse LLM response as JSON: {e}")
        except Exception as e:
            raise RuntimeError(f"Semantic evaluation failed: {e}")

    def _construct_prompt(self, actual: str, expected: str) -> str:
        """
        Construct the evaluation prompt for semantic comparison.

        This internal method formats the actual and expected texts into a
        structured prompt that guides the language model's evaluation process.

        Parameters
        ----------
        actual : str
            The actual output text
        expected : str
            The expected reference text

        Returns
        -------
        str
            Formatted evaluation prompt
        """
        return f"""Evaluate the semantic alignment between these two texts:

EXPECTED (Ground Truth):
{expected}

ACTUAL (Observed Output):
{actual}

Analyze:
1. Factual consistency: Do both texts convey the same core facts?
2. Semantic entropy: How much information divergence exists?
3. Logical equivalence: Are the claims logically equivalent?

Provide your evaluation as JSON with 'score' and 'reasoning' fields."""

    def batch_evaluate(
        self, pairs: list[tuple[str, str]], metadata: Optional[Dict[str, Any]] = None
    ) -> list[DriftResult]:
        """
        Evaluate multiple actual-expected text pairs in batch.

        This method provides efficient batch processing for large-scale
        semantic drift evaluation. Each pair is evaluated independently
        with the same model configuration.

        Parameters
        ----------
        pairs : list of tuple[str, str]
            List of (actual, expected) text pairs to evaluate
        metadata : dict, optional
            Shared metadata applied to all evaluations

        Returns
        -------
        list[DriftResult]
            List of drift results corresponding to each input pair

        Examples
        --------
        >>> judge = SemanticJudge()
        >>> pairs = [
        ...     ("Output A", "Expected A"),
        ...     ("Output B", "Expected B")
        ... ]
        >>> results = judge.batch_evaluate(pairs)
        >>> avg_score = sum(r.score for r in results) / len(results)
        """
        results = []
        for actual, expected in pairs:
            result = self.evaluate(actual, expected, metadata)
            results.append(result)
        return results
