# Contributing to Semantic Unit

## Governance Model

Semantic Unit is maintained as a research artifact under strict academic standards for code quality, methodological rigor, and semantic integrity. All contributions undergo comprehensive technical and conceptual review to ensure alignment with the framework's foundational principles of deterministic evaluation and reproducible science.

## Maintainer

**Principal Maintainer**: [Your Name]  
**Affiliation**: [Your Institution/Independent Researcher]  
**Contact**: your.email@example.com  
**ORCID**: [Your ORCID if available]

The Principal Maintainer holds final authority on all technical and conceptual decisions, ensuring the framework maintains its research integrity and methodological consistency with published evaluation standards.

## Contribution Philosophy

This project adheres to the highest standards of academic software engineering. Contributions must demonstrate:

1. **Semantic Integrity**: All evaluation logic must preserve deterministic properties and semantic consistency
2. **Methodological Rigor**: Changes must be grounded in established evaluation theory or novel empirical findings
3. **Reproducibility**: Code must enable exact reproduction of results across environments
4. **Documentation Excellence**: Academic-grade documentation with formal terminology and citations where applicable

## Submission Guidelines

### Before Submitting

1. **Review the Literature**: Familiarize yourself with relevant publications on LLM evaluation, semantic drift detection, and deterministic testing methodologies
2. **Understand the Architecture**: Study the core evaluation engine and its theoretical foundations
3. **Run the Test Suite**: Ensure all existing tests pass (`pytest`)
4. **Validate Type Safety**: Run static type checking (`mypy semantic_unit`)
5. **Format Your Code**: Apply standardized formatting (`black semantic_unit`, `ruff check semantic_unit`)

### Pull Request Process

All Pull Requests undergo a **Semantic Integrity Review** conducted by the Principal Maintainer. This review evaluates:

#### Technical Criteria
- **Correctness**: Does the implementation correctly solve the stated problem?
- **Determinism**: Does the change preserve reproducibility guarantees?
- **Performance**: Are computational resources used efficiently?
- **Test Coverage**: Are changes adequately covered by unit and integration tests?
- **Type Safety**: Are all functions properly annotated with type hints?

#### Semantic Criteria
- **Conceptual Soundness**: Is the approach theoretically justified?
- **Semantic Preservation**: Do changes maintain factual and logical consistency?
- **Alignment with Framework Principles**: Does the contribution align with deterministic evaluation goals?
- **Nomenclature Consistency**: Is formal academic terminology used appropriately?

#### Documentation Criteria
- **Docstring Completeness**: Are all public APIs documented in Google/NumPy style?
- **Mathematical Notation**: Are algorithms explained with proper mathematical formalism where applicable?
- **Example Quality**: Do examples demonstrate best practices and proper usage patterns?
- **Citation Accuracy**: Are external methods and algorithms properly attributed?

### Review Timeline

- **Initial Triage**: Within 72 hours
- **Technical Review**: 1-2 weeks depending on complexity
- **Revision Cycles**: Contributors should respond to feedback within 1 week
- **Final Decision**: Maintainer provides accept/reject decision with detailed justification

### Acceptance Criteria

Pull Requests are accepted when they:

1. ✅ Pass all automated tests and quality checks
2. ✅ Demonstrate clear improvements to functionality, performance, or documentation
3. ✅ Maintain semantic integrity and deterministic properties
4. ✅ Include comprehensive tests covering edge cases
5. ✅ Provide academic-quality documentation
6. ✅ Receive explicit approval from the Principal Maintainer

## Types of Contributions

### High-Priority Contributions

- **Novel Evaluation Metrics**: New semantic similarity measures grounded in published research
- **Performance Optimizations**: Improvements to computational efficiency with benchmarks
- **Expanded LLM Support**: Integration of additional language models with validation studies
- **Empirical Validation**: Benchmark datasets and comparative studies against baseline methods

### Standard Contributions

- **Bug Fixes**: Corrections to incorrect behavior with test cases demonstrating the issue
- **Documentation Improvements**: Clarifications, examples, and theoretical explanations
- **Test Coverage**: Additional test cases improving edge case handling
- **Type Hint Enhancements**: Improved static type safety

### Special Approval Required

The following require explicit design discussion before implementation:

- Changes to core evaluation algorithms
- Modifications to the semantic alignment scoring methodology
- New dependencies that may affect reproducibility
- Breaking changes to the public API
- Alternative evaluation paradigms

## Code Standards

### Style and Formatting

- **Python Version**: Code must support Python 3.9+
- **Formatting**: Black (line length: 100 characters)
- **Linting**: Ruff with full error checking
- **Type Checking**: Mypy in strict mode
- **Import Sorting**: Maintained automatically by Ruff

### Documentation Standards

All public functions, classes, and modules must include:

```python
def evaluate(self, actual: str, expected: str) -> DriftResult:
    """
    Evaluate semantic drift between actual and expected outputs.
    
    This method quantifies distributional divergence in the latent
    semantic space using zero-shot LLM-based comparison.
    
    Parameters
    ----------
    actual : str
        Observed output from the system under evaluation
    expected : str
        Reference ground truth for comparison
        
    Returns
    -------
    DriftResult
        Structured evaluation containing alignment score and reasoning
        
    Raises
    ------
    ValueError
        If inputs are empty or invalid
        
    Examples
    --------
    >>> judge = SemanticJudge()
    >>> result = judge.evaluate("Output A", "Expected A")
    >>> print(f"Score: {result.score}")
    
    Notes
    -----
    Evaluation uses temperature=0.0 for deterministic results.
    Minor variations may occur due to model version updates.
    
    References
    ----------
    .. [1] Liu et al. (2023). "G-Eval: NLG Evaluation..."
    """
```

### Testing Standards

- **Unit Test Coverage**: Minimum 85% coverage for new code
- **Integration Tests**: Required for API changes
- **Doctest Examples**: All examples in docstrings must be executable
- **Parametrized Tests**: Use `pytest.mark.parametrize` for multiple test cases
- **Mocking**: Mock external API calls (LiteLLM) in unit tests

### Commit Message Format

Use conventional commit format:

```
type(scope): Brief description

Detailed explanation of changes, including:
- Motivation for the change
- Implementation approach
- Any breaking changes or migration notes

References: #issue-number
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

## Research Collaboration

### Academic Use

Researchers using Semantic Unit in publications should:

1. Cite using the provided `CITATION.cff` file
2. Report version numbers for reproducibility
3. Share configuration details (model, temperature, etc.)
4. Acknowledge the framework in papers and presentations

### Collaborative Research

For collaborative research projects involving framework extensions:

1. Contact the Principal Maintainer with a research proposal
2. Establish a collaboration agreement defining contributions and authorship
3. Follow journal-specific guidelines for software contributions
4. Co-author documentation for novel methodological contributions

## Code of Conduct

### Scientific Integrity

- Report results honestly without selective reporting or p-hacking
- Acknowledge limitations and potential biases in evaluation methods
- Provide reproducible examples and clear documentation
- Respect intellectual property and properly attribute prior work

### Community Standards

- Engage in constructive, respectful technical discussions
- Provide actionable, specific feedback in code reviews
- Welcome contributors of all experience levels
- Focus on technical merit rather than personal preferences

### Prohibited Conduct

- Plagiarism or inadequate attribution of others' work
- Submission of low-quality code without proper testing
- Circumventing the review process
- Harassing or unprofessional behavior toward maintainers or contributors

## Recognition

### Contributor Recognition

Accepted contributions are recognized through:

1. **GitHub Contribution Graph**: Automatic tracking of commits
2. **Release Notes**: Credit in version release announcements  
3. **Contributors File**: Listed in project documentation
4. **Research Acknowledgments**: Significant contributions acknowledged in papers

### Authorship Criteria

Substantial intellectual contributions may warrant co-authorship on research papers describing the framework. Authorship is determined using [ICMJE criteria](http://www.icmje.org/recommendations/browse/roles-and-responsibilities/defining-the-role-of-authors-and-contributors.html):

- Substantial contributions to design, implementation, or validation
- Critical manuscript revision for intellectual content
- Final approval of published version
- Accountability for all aspects of the work

## Questions and Support

### Technical Questions

- **GitHub Discussions**: https://github.com/chaitanyakumar-d/semantic-unit/discussions
- **GitHub Issues**: https://github.com/chaitanyakumar-d/semantic-unit/issues
- **Email**: Direct maintainer contact for sensitive issues

### Research Inquiries

For research collaborations, benchmark requests, or methodological questions, contact the Principal Maintainer directly with:

- Brief description of your research
- Specific questions or collaboration proposals
- Timeline and deliverable expectations

## License

By contributing to Semantic Unit, you agree that your contributions will be licensed under the MIT License. You represent that you have the legal right to make your contributions.

## Acknowledgments

We acknowledge the broader research community working on LLM evaluation, including the authors of G-Eval, MT-Bench, and other foundational evaluation frameworks that inspire this work.

---

**Last Updated**: January 7, 2026  
**Version**: 1.0  
**Principal Maintainer**: [Your Name]
