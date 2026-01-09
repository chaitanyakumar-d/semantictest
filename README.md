# Semantic Unit: Unit Testing for AI Agents

[![Tests](https://github.com/chaitanyakumar-d/semantic-unit/actions/workflows/tests.yml/badge.svg)](https://github.com/chaitanyakumar-d/semantic-unit/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub stars](https://img.shields.io/github/stars/chaitanyakumar-d/semantic-unit?style=social)](https://github.com/chaitanyakumar-d/semantic-unit)

> **Stop your AI from hallucinating in production with one line of code.**

A modern testing framework that brings unit testing to AI agents. Test for **meaning**, not syntax. Catch AI hallucinations before they reach production.

## The Problem

Traditional testing breaks with AI:

```python
# ❌ This fails even when AI is correct
assert ai_response == "The test passed successfully"
# AI says: "The test was successful" → TEST FAILS (but meaning is correct!)
```

AI outputs are **never** identical, even when correct. Your tests shouldn't break on paraphrasing.

## The Solution

```python
from semantic_unit import SemanticJudge

judge = SemanticJudge()
result = judge.evaluate(
    actual=ai_response,
    expected="The test passed successfully"
)

assert result.score > 0.8  # ✅ Tests meaning, not exact words
```

## Why Semantic Unit?

- 🎯 **Test Meaning, Not Words**: Assert on semantic correctness, not string equality
- 🛡️ **Prevent Hallucinations**: Catch AI drift before it reaches users
- ⚡ **Drop-in Replacement**: Works with pytest, unittest, or any test framework
- 🔬 **Deterministic**: Reproducible results for reliable CI/CD
- 📊 **Actionable Insights**: Get explanations for why tests pass/fail
- 🚀 **Production Ready**: Battle-tested with comprehensive test coverage

## Installation

```bash
pip install semantic-unit
```

Or install from source:

```bash
git clone https://github.com/chaitanyakumar-d/semantic-unit.git
cd semantic-unit
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

**1. Testing AI Responses:**

```python
from semantic_unit import SemanticJudge

judge = SemanticJudge()

# Your AI agent's output
ai_output = "The experiment succeeded with 95% accuracy"
expected = "The experiment achieved 95% accuracy"

# Test semantic correctness
result = judge.evaluate(ai_output, expected)

if result.score > 0.8:
    print("✓ AI response is correct")
else:
    print(f"✗ AI drifted: {result.reasoning}")
```

**2. Integration with pytest:**

```python
import pytest
from semantic_unit import SemanticJudge

@pytest.fixture
def judge():
    return SemanticJudge()

def test_ai_customer_support(judge):
    ai_response = get_ai_response("What's your return policy?")
    expected = "We accept returns within 30 days"

    result = judge.evaluate(ai_response, expected)
    assert result.score > 0.8, f"AI hallucinated: {result.reasoning}"

def test_ai_summarization(judge):
    summary = ai_summarize(long_document)
    expected_points = "Revenue increased, costs decreased, profit margins improved"

    result = judge.evaluate(summary, expected_points)
    assert result.score > 0.7, "Summary missing key points"
```

**3. CLI Usage:**

```bash
# Quick evaluation
semantic-unit evaluate "AI said this" "Should mean this"

# Batch testing
semantic-unit batch test_cases.json --output results.json

# With custom model
semantic-unit evaluate "text" "expected" --model gpt-4
```

## Real-World Use Cases

### 1. **AI Chatbot Testing**
```python
# Test customer support AI doesn't hallucinate policies
result = judge.evaluate(
    actual=chatbot_response,
    expected="We offer 24/7 support with 1-hour response time"
)
assert result.score > 0.9, "Chatbot gave wrong information!"
```

### 2. **RAG Pipeline Validation**
```python
# Ensure retrieval-augmented generation stays accurate
result = judge.evaluate(
    actual=rag_output,
    expected=ground_truth_answer
)
assert result.score > 0.85, "RAG hallucinated facts"
```

### 3. **AI Agent Monitoring**
```python
# Production monitoring for AI drift
for ai_response in production_logs:
    result = judge.evaluate(ai_response, expected_behavior)
    if result.score < 0.7:
        alert_team(f"AI drift detected: {result.reasoning}")
```

### 4. **Fine-tuning Validation**
```python
# Verify fine-tuned model maintains accuracy
test_cases = load_test_suite()
results = judge.batch_evaluate(test_cases)
avg_score = sum(r.score for r in results) / len(results)
assert avg_score > 0.8, "Fine-tuning degraded performance"
```

## Who Uses Semantic Unit?

- **AI Engineers**: Testing LLM applications and agents
- **QA Teams**: Automated testing of AI features
- **DevOps**: Monitoring AI systems in production
- **Researchers**: Evaluating model performance
- **Startups**: Shipping AI products with confidence

## Configuration

### API Keys

Create a `.env` file:

```bash
# OpenAI (default)
OPENAI_API_KEY=your-openai-key

# Or use other providers:
ANTHROPIC_API_KEY=your-anthropic-key
GEMINI_API_KEY=your-google-key
GROQ_API_KEY=your-groq-key
AZURE_API_KEY=your-azure-key
AZURE_API_BASE=https://your-resource.openai.azure.com
```

### Supported LLM Providers

Semantic Unit supports **100+ LLMs** via LiteLLM:

| Provider | Models | Environment Variable |
|----------|--------|---------------------|
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` | `OPENAI_API_KEY` |
| **Anthropic** | `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229` | `ANTHROPIC_API_KEY` |
| **Google** | `gemini/gemini-1.5-pro`, `gemini/gemini-1.5-flash` | `GEMINI_API_KEY` |
| **Azure** | `azure/gpt-4o`, `azure/gpt-4` | `AZURE_API_KEY` + `AZURE_API_BASE` |
| **Groq** | `groq/llama-3.1-70b-versatile`, `groq/mixtral-8x7b-32768` | `GROQ_API_KEY` |
| **Ollama** | `ollama/llama3.1`, `ollama/mistral` | Local (no key needed) |
| **AWS Bedrock** | `bedrock/anthropic.claude-3-sonnet` | AWS credentials |
| **Together AI** | `together_ai/meta-llama/Llama-3-70b-chat-hf` | `TOGETHER_API_KEY` |

### Usage Examples

```python
from semantic_unit import SemanticJudge

# OpenAI (default)
judge = SemanticJudge(model="gpt-4o-mini")

# Anthropic Claude
judge = SemanticJudge(model="claude-3-5-sonnet-20241022")

# Google Gemini
judge = SemanticJudge(model="gemini/gemini-1.5-flash")

# Groq (fast inference)
judge = SemanticJudge(model="groq/llama-3.1-70b-versatile")

# Local Ollama (free, private)
judge = SemanticJudge(model="ollama/llama3.1")

# Azure OpenAI
judge = SemanticJudge(model="azure/gpt-4o")

# List all supported models
print(SemanticJudge.list_supported_models())
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/chaitanyakumar-d/semantic-unit.git
cd semantic-unit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=semantic_unit

# Run specific test file
pytest tests/test_core.py
```

### Code Quality

```bash
# Format code
black semantic_unit tests

# Lint code
ruff check semantic_unit tests

# Type checking
mypy semantic_unit
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Why This Matters

**AI is everywhere, but testing AI is broken.**

- Traditional tests: `assert output == expected` ❌
- Semantic Unit: `assert meaning_matches(output, expected)` ✅

**The difference?** Your AI can now:
- Paraphrase freely without breaking tests
- Improve responses without false failures
- Scale to production with confidence

## Repository & Links

- **GitHub**: https://github.com/chaitanyakumar-d/semantic-unit
- **Issues**: https://github.com/chaitanyakumar-d/semantic-unit/issues
- **Discussions**: https://github.com/chaitanyakumar-d/semantic-unit/discussions

## Roadmap

- [x] Core semantic evaluation engine
- [x] CLI with evaluate and batch commands
- [x] Comprehensive test suite (20+ tests)
- [x] CI/CD with GitHub Actions
- [ ] PyPI publication
- [ ] Additional evaluation metrics
- [ ] Performance benchmarks
- [ ] Documentation website
- [ ] Integration examples (LangChain, LlamaIndex, etc.)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
