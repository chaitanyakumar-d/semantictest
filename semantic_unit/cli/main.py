"""
Main CLI entry point for Semantic Unit.

This module defines the primary CLI application using Typer.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from semantic_unit import SemanticJudge, DriftResult

app = typer.Typer(
    name="semantic-unit",
    help="Semantic Unit: Deterministic Evaluation Standard",
    add_completion=False,
)
console = Console()


def version_callback(value: bool) -> None:
    """Display version information."""
    if value:
        from semantic_unit import __version__

        console.print(f"Semantic Unit version: {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """
    Semantic Unit: Deterministic Evaluation Standard

    A modern framework for evaluating semantic units with deterministic,
    reproducible results.
    """
    pass


@app.command()
def evaluate(
    actual: str = typer.Argument(..., help="Actual output text to evaluate"),
    expected: str = typer.Argument(..., help="Expected reference text"),
    model: str = typer.Option(
        "gpt-4o-mini", "--model", "-m", help="LLM model to use for evaluation"
    ),
    temperature: float = typer.Option(
        0.0, "--temperature", "-t", help="Sampling temperature (0.0 for deterministic)"
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to JSON file"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
) -> None:
    """
    Evaluate semantic drift between actual and expected text.

    This command performs LLM-based semantic comparison to measure alignment
    and detect distributional drift between observed and reference outputs.

    Examples:
        semantic-unit evaluate "The test passed" "Test was successful"
        semantic-unit evaluate "Output A" "Expected A" --model gpt-4 --json
    """
    try:
        # Initialize semantic judge
        judge = SemanticJudge(model=model, temperature=temperature)

        # Perform evaluation
        with console.status("[bold green]Evaluating semantic alignment...", spinner="dots"):
            result = judge.evaluate(actual=actual, expected=expected)

        # Output results
        if json_output:
            output_data = {
                "score": result.score,
                "reasoning": result.reasoning,
                "actual": result.actual,
                "expected": result.expected,
                "model": result.model,
                "metadata": result.metadata,
            }
            console.print(json.dumps(output_data, indent=2))
        else:
            # Rich formatted output
            console.print()
            console.print(
                Panel.fit(
                    f"[bold cyan]Semantic Alignment Score:[/bold cyan] [bold yellow]{result.score:.3f}[/bold yellow]",
                    border_style="green",
                )
            )

            console.print("\n[bold]Reasoning:[/bold]")
            console.print(f"  {result.reasoning}\n")

            table = Table(title="Evaluation Details", show_header=True)
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="white")

            table.add_row(
                "Actual", result.actual[:100] + "..." if len(result.actual) > 100 else result.actual
            )
            table.add_row(
                "Expected",
                result.expected[:100] + "..." if len(result.expected) > 100 else result.expected,
            )
            table.add_row("Model", result.model)
            if result.metadata and "tokens_used" in result.metadata:
                table.add_row("Tokens Used", str(result.metadata["tokens_used"]))

            console.print(table)
            console.print()

        # Save to file if requested
        if output:
            output_data = result.model_dump()
            output.write_text(json.dumps(output_data, indent=2))
            console.print(f"[green]✓[/green] Results saved to {output}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}", style="red")
        raise typer.Exit(code=1)


@app.command()
def batch(
    input_file: Path = typer.Argument(..., help="JSON file with array of {actual, expected} pairs"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="LLM model to use"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to JSON file"),
) -> None:
    """
    Evaluate multiple text pairs from a JSON file.

    Input file format:
    [
        {"actual": "text1", "expected": "ref1"},
        {"actual": "text2", "expected": "ref2"}
    ]

    Example:
        semantic-unit batch evaluations.json --output results.json
    """
    try:
        # Load input file
        if not input_file.exists():
            console.print(f"[red]Error:[/red] File not found: {input_file}")
            raise typer.Exit(code=1)

        with open(input_file) as f:
            pairs_data = json.load(f)

        if not isinstance(pairs_data, list):
            console.print("[red]Error:[/red] Input file must contain a JSON array")
            raise typer.Exit(code=1)

        # Extract pairs
        pairs = [(item["actual"], item["expected"]) for item in pairs_data]

        # Initialize judge
        judge = SemanticJudge(model=model)

        # Evaluate batch
        console.print(f"[bold]Evaluating {len(pairs)} pairs...[/bold]\n")
        results = []

        with console.status("[bold green]Processing...", spinner="dots"):
            for i, (actual, expected) in enumerate(pairs, 1):
                result = judge.evaluate(actual, expected)
                results.append(result)
                console.print(f"  [{i}/{len(pairs)}] Score: {result.score:.3f}")

        # Calculate statistics
        avg_score = sum(r.score for r in results) / len(results)
        min_score = min(r.score for r in results)
        max_score = max(r.score for r in results)

        console.print(f"\n[bold cyan]Summary Statistics:[/bold cyan]")
        console.print(f"  Average Score: {avg_score:.3f}")
        console.print(f"  Min Score: {min_score:.3f}")
        console.print(f"  Max Score: {max_score:.3f}")

        # Save results
        if output:
            output_data = [r.model_dump() for r in results]
            output.write_text(json.dumps(output_data, indent=2))
            console.print(f"\n[green]✓[/green] Results saved to {output}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}", style="red")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
