"""CLI interface for chunker optimizer"""
import click
import json
from pathlib import Path
from typing import Optional

try:
    # Try absolute imports first (for module execution)
    from chunker_optimizer.domain.entities.prompt import Prompt
    from chunker_optimizer.domain.entities.chunking_context import ChunkingContext
    from chunker_optimizer.domain.entities.chunk import Chunk
    from chunker_optimizer.application.use_cases.run_optimization_loop import (
        RunOptimizationLoopUseCase,
        OptimizationConfig,
        OptimizationResult
    )
    from chunker_optimizer.application.use_cases.evaluate_chunks import EvaluateChunksUseCase
    from chunker_optimizer.application.use_cases.optimize_prompt import OptimizePromptUseCase
    from chunker_optimizer.infrastructure.llm.openai_client import OpenAIClient
    from chunker_optimizer.infrastructure.performance.metrics_collector import PerformanceMetricsCollector
    from chunker_optimizer.infrastructure.performance.profiler import OptimizationProfiler
    from chunker_optimizer.domain.entities.document_context import (
        DocumentEnrichment,
        ParsedDocument,
        VLMOutput,
        LLMSummary
    )
except ImportError:
    # Fallback to relative imports (for package structure)
    from ...domain.entities.prompt import Prompt
    from ...domain.entities.chunking_context import ChunkingContext
    from ...domain.entities.chunk import Chunk
    from ...application.use_cases.run_optimization_loop import (
        RunOptimizationLoopUseCase,
        OptimizationConfig,
        OptimizationResult
    )
    from ...application.use_cases.evaluate_chunks import EvaluateChunksUseCase
    from ...application.use_cases.optimize_prompt import OptimizePromptUseCase
    from ...infrastructure.llm.openai_client import OpenAIClient
    from ...infrastructure.performance.metrics_collector import PerformanceMetricsCollector
    from ...infrastructure.performance.profiler import OptimizationProfiler
    from ...domain.entities.document_context import (
        DocumentEnrichment,
        ParsedDocument,
        VLMOutput,
        LLMSummary
    )


def create_chunking_function_from_context(chunking_context: ChunkingContext) -> list[Chunk]:
    """
    Create a simple chunking function for testing
    
    This is a placeholder - in production, this would call the actual chunking pipeline
    """
    # Simple chunking: split by paragraphs
    text = chunking_context.document_enrichment.parsed_document.raw_content
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    chunks = []
    current_index = 0
    for i, para in enumerate(paragraphs):
        start_index = current_index
        end_index = start_index + len(para)
        chunks.append(Chunk(
            id=f"chunk_{i}",
            content=para,
            start_index=start_index,
            end_index=end_index
        ))
        current_index = end_index + 2  # +2 for "\n\n"
    
    return chunks


def load_real_data(chunk_unit: str, filename: Optional[str] = None) -> Optional[DocumentEnrichment]:
    """
    Load real enrichment data (optional import from tests)
    
    Args:
        chunk_unit: Chunk unit type
        filename: Optional specific filename
    
    Returns:
        DocumentEnrichment or None
    """
    try:
        # Try to import and use real data loader
        import sys
        from pathlib import Path
        tests_path = Path(__file__).parent.parent.parent.parent / "tests"
        if tests_path.exists():
            sys.path.insert(0, str(tests_path.parent))
            from tests.fixtures.real_data_loader import RealEnrichmentDataLoader
        
        if filename:
            return RealEnrichmentDataLoader.load_from_json(filename)
        else:
            return RealEnrichmentDataLoader.load_by_chunk_unit(chunk_unit)
    except (ImportError, FileNotFoundError):
        return None


def create_mock_enrichment(chunk_unit: str, document_text: Optional[str] = None) -> DocumentEnrichment:
    """
    Create mock enrichment data
    
    Args:
        chunk_unit: Chunk unit type
        document_text: Optional document text
    
    Returns:
        DocumentEnrichment
    """
    if document_text is None:
        document_text = "This is a sample document for testing chunking optimization."
    
    parsed_doc = ParsedDocument(
        raw_content=document_text,
        parsed_elements=[{"type": "paragraph", "content": document_text, "index": 0}],
        metadata={"source": "cli_mock"}
    )
    
    vlm_outputs = []
    if chunk_unit == "page":
        vlm_outputs = [
            VLMOutput(
                page_id="page_1",
                script="Mock VLM script describing the page layout.",
                visual_elements=[],
                metadata={"mock": True}
            )
        ]
    
    llm_summary = LLMSummary(
        summary="Mock LLM summary of the document.",
        key_points=["Point 1", "Point 2"],
        topics=["Topic A"],
        structure_info={"sections": 1},
        metadata={"mock": True}
    )
    
    return DocumentEnrichment(
        parsed_document=parsed_doc,
        vlm_outputs=vlm_outputs,
        llm_summary=llm_summary,
        enrichment_metadata={"chunk_unit": chunk_unit}
    )


@click.group()
def cli():
    """LLM Chunker Optimizer CLI"""
    pass


@cli.command()
@click.option("--document", help="Document file path or text")
@click.option("--initial-prompt", required=True, help="Initial chunking prompt")
@click.option("--chunk-unit", default="page", type=click.Choice(["page", "element", "lifelog", "default"]), help="Chunk unit")
@click.option("--threshold", default=0.8, type=float, help="Quality threshold (0.0 to 1.0)")
@click.option("--max-iterations", default=10, type=int, help="Maximum iterations")
@click.option("--use-real-data", is_flag=True, help="Use real enrichment data from JSON")
@click.option("--real-data-file", help="Specific real data JSON file to use")
@click.option("--output", help="Output file path for results (JSON format)")
@click.option("--enable-profiling", is_flag=True, help="Enable performance profiling")
def optimize(
    document: Optional[str],
    initial_prompt: str,
    chunk_unit: str,
    threshold: float,
    max_iterations: int,
    use_real_data: bool,
    real_data_file: Optional[str],
    output: Optional[str],
    enable_profiling: bool
):
    """
    Optimize chunking prompt based on evaluation metrics
    
    Example:
        python -m chunker_optimizer.presentation.cli optimize \\
            --initial-prompt "Split text into meaningful chunks" \\
            --chunk-unit page \\
            --threshold 0.8 \\
            --max-iterations 10
    """
    click.echo("🚀 Starting chunking optimization...")
    
    # Load enrichment data
    if use_real_data:
        click.echo("📂 Loading real enrichment data...")
        enrichment = load_real_data(chunk_unit, real_data_file)
        if enrichment is None:
            click.echo(f"⚠️  No real data found for chunk_unit={chunk_unit}, using mock data", err=True)
            enrichment = create_mock_enrichment(chunk_unit, document)
    else:
        click.echo("📝 Using mock enrichment data...")
        enrichment = create_mock_enrichment(chunk_unit, document)
    
    # Create chunking context
    prompt = Prompt(id="cli_optimization", content=initial_prompt, version=1)
    context = ChunkingContext(
        document_enrichment=enrichment,
        prompt=prompt,
        chunk_unit=chunk_unit
    )
    
    # Initialize use cases
    click.echo("🔧 Initializing components...")
    evaluate_use_case = EvaluateChunksUseCase()
    
    # Initialize LLM client (optional - can work without it for testing)
    try:
        llm_client = OpenAIClient()
        optimize_use_case = OptimizePromptUseCase(llm_client=llm_client)
        click.echo("✅ LLM client initialized")
    except Exception as e:
        click.echo(f"⚠️  LLM client not available: {e}", err=True)
        click.echo("⚠️  Using mock LLM client for testing", err=True)
        from ...infrastructure.llm.llm_client import LLMClient
        
        class MockLLMClient(LLMClient):
            def optimize_prompt(self, current_prompt: str, optimization_instruction: str) -> str:
                return f"Optimized: {current_prompt}"
        
        optimize_use_case = OptimizePromptUseCase(llm_client=MockLLMClient())
    
    # Initialize performance monitoring
    metrics_collector = PerformanceMetricsCollector()
    profiler = OptimizationProfiler(enable=enable_profiling)
    
    if enable_profiling:
        profiler.start()
    
    # Create chunking function
    chunking_function = create_chunking_function_from_context
    
    # Create optimization use case
    use_case = RunOptimizationLoopUseCase(
        evaluate_use_case=evaluate_use_case,
        optimize_use_case=optimize_use_case,
        chunking_function=chunking_function,
        metrics_collector=metrics_collector
    )
    
    # Run optimization
    config = OptimizationConfig(
        max_iterations=max_iterations,
        threshold=threshold
    )
    
    click.echo(f"\n📊 Configuration:")
    click.echo(f"   - Max iterations: {max_iterations}")
    click.echo(f"   - Threshold: {threshold}")
    click.echo(f"   - Chunk unit: {chunk_unit}")
    click.echo(f"   - Using real data: {use_real_data}")
    click.echo("\n🔄 Running optimization loop...\n")
    
    result = use_case.execute(prompt, context, config)
    
    if enable_profiling:
        profiler.stop()
    
    # Display results
    click.echo("\n" + "="*60)
    click.echo("✅ Optimization Complete!")
    click.echo("="*60)
    click.echo(f"\n📈 Results:")
    click.echo(f"   - Iterations: {result.iterations}")
    click.echo(f"   - Converged: {'Yes' if result.converged else 'No'}")
    click.echo(f"   - Final Score: {result.final_metrics.overall_score:.3f}")
    click.echo(f"   - Boundary Clarity: {result.final_metrics.boundary_clarity:.3f}")
    click.echo(f"   - Chunk Stickiness: {result.final_metrics.chunk_stickiness:.3f}")
    click.echo(f"   - HOPE Score: {result.final_metrics.hope_score:.3f}")
    
    # Performance metrics
    performance = metrics_collector.get_summary()
    if performance.total_iterations > 0:
        click.echo(f"\n⏱️  Performance:")
        click.echo(f"   - Total time: {performance.total_time:.2f}s")
        click.echo(f"   - Avg evaluation time: {performance.average_evaluation_time:.3f}s")
        click.echo(f"   - Avg optimization time: {performance.average_optimization_time:.3f}s")
        click.echo(f"   - Avg chunking time: {performance.average_chunking_time:.3f}s")
    
    click.echo(f"\n📝 Final Prompt (v{result.final_prompt.version}):")
    click.echo(f"   {result.final_prompt.content}")
    
    # Save results if output specified
    if output:
        output_path = Path(output)
        output_data = {
            "result": {
                "iterations": result.iterations,
                "converged": result.converged,
                "final_metrics": result.final_metrics.to_dict(),
                "final_prompt": {
                    "id": result.final_prompt.id,
                    "content": result.final_prompt.content,
                    "version": result.final_prompt.version
                }
            },
            "history": result.history,
            "performance": {
                "total_time": performance.total_time,
                "average_times": {
                    "evaluation": performance.average_evaluation_time,
                    "optimization": performance.average_optimization_time,
                    "chunking": performance.average_chunking_time
                }
            }
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        click.echo(f"\n💾 Results saved to: {output_path}")
    
    # Save profiling report if enabled
    if enable_profiling:
        profile_path = Path("profile_report.txt")
        profiler.save_report(profile_path)
        click.echo(f"📊 Profiling report saved to: {profile_path}")


@cli.command()
@click.option("--test-cases", help="Path to test cases JSON file")
@click.option("--iterations", default=3, help="Number of benchmark iterations")
@click.option("--output", help="Output file for benchmark report")
def benchmark(test_cases: Optional[str], iterations: int, output: Optional[str]):
    """
    Run performance benchmarks
    
    Example:
        python -m chunker_optimizer.presentation.cli benchmark \\
            --iterations 5 \\
            --output benchmark_report.json
    """
    click.echo("📊 Running benchmarks...")
    click.echo("⚠️  Benchmark functionality requires test cases configuration")
    click.echo("   This feature will be enhanced in future versions")


if __name__ == "__main__":
    cli()
