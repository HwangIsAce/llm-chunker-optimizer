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
    # Try to import peter-parser adapter
    try:
        from chunker_optimizer.infrastructure.adapters.peter_parser_adapter import (
            PeterParserAdapter,
            create_peter_parser_chunking_function
        )
        PETER_PARSER_AVAILABLE = True
    except ImportError:
        PETER_PARSER_AVAILABLE = False
        PeterParserAdapter = None
        create_peter_parser_chunking_function = None
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
    # Try to import peter-parser adapter
    try:
        from ...infrastructure.adapters.peter_parser_adapter import (
            PeterParserAdapter,
            create_peter_parser_chunking_function
        )
        PETER_PARSER_AVAILABLE = True
    except ImportError:
        PETER_PARSER_AVAILABLE = False
        PeterParserAdapter = None
        create_peter_parser_chunking_function = None


def create_chunking_function_from_context(chunking_context: ChunkingContext) -> list[Chunk]:
    """
    Create a chunking function that uses the prompt from chunking context
    
    The prompt is analyzed to determine chunking strategy:
    - If prompt emphasizes "semantic boundaries" or "coherent", use semantic-based chunking
    - If prompt emphasizes "distinct" or "standalone", create more distinct chunks
    - If prompt emphasizes "boundaries", focus on clear boundary detection
    - Otherwise, use adaptive chunking based on prompt keywords
    """
    text = chunking_context.document_enrichment.parsed_document.raw_content
    prompt_content = chunking_context.prompt.content.lower()
    
    # Analyze prompt to determine chunking strategy
    use_semantic = any(keyword in prompt_content for keyword in [
        "semantic", "coherent", "coherence", "meaningful", "theme"
    ])
    use_distinct = any(keyword in prompt_content for keyword in [
        "distinct", "standalone", "independent", "separate", "unique"
    ])
    use_boundaries = any(keyword in prompt_content for keyword in [
        "boundary", "boundaries", "boundary marker", "sharply defined"
    ])
    use_topic_shift = any(keyword in prompt_content for keyword in [
        "topic shift", "topic change", "new topic", "unique topic"
    ])
    
    # Determine chunk size based on prompt emphasis
    if use_distinct:
        # Smaller, more distinct chunks
        chunk_size_factor = 0.7
    elif use_semantic or use_topic_shift:
        # Medium-sized semantic chunks
        chunk_size_factor = 1.0
    else:
        # Default chunking
        chunk_size_factor = 1.2
    
    # Split text into potential chunks
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    # If no paragraphs, split by sentences
    if not paragraphs:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if sentences:
            paragraphs = sentences
        else:
            # Fallback: split by newlines
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    
    # Apply chunking strategy
    chunks = []
    current_index = 0
    
    if use_semantic or use_topic_shift:
        # Group paragraphs by semantic similarity (simplified)
        # In production, this would use embeddings
        chunk_groups = []
        current_group = []
        target_size = int(len(paragraphs) * chunk_size_factor / max(1, len(paragraphs) // 10))
        
        for i, para in enumerate(paragraphs):
            current_group.append((i, para))
            # Create chunk when group reaches target size or at topic shift indicators
            if len(current_group) >= target_size or (use_topic_shift and i > 0 and i % 3 == 0):
                chunk_groups.append(current_group)
                current_group = []
        if current_group:
            chunk_groups.append(current_group)
        
        # Create chunks from groups
        for group_idx, group in enumerate(chunk_groups):
            chunk_text = "\n\n".join([para for _, para in group])
            start_idx = current_index
            end_idx = start_idx + len(chunk_text)
            chunks.append(Chunk(
                id=f"chunk_{group_idx}",
                content=chunk_text,
                start_index=start_idx,
                end_index=end_idx
            ))
            current_index = end_idx + 2
    elif use_distinct:
        # Create more distinct, smaller chunks
        for i, para in enumerate(paragraphs):
            # Split long paragraphs further if needed
            if len(para) > 500:  # Split long paragraphs
                sentences = [s.strip() for s in para.split(".") if s.strip()]
                for j, sentence in enumerate(sentences):
                    start_idx = current_index
                    end_idx = start_idx + len(sentence)
                    chunks.append(Chunk(
                        id=f"chunk_{i}_{j}",
                        content=sentence,
                        start_index=start_idx,
                        end_index=end_idx
                    ))
                    current_index = end_idx + 2
            else:
                start_idx = current_index
                end_idx = start_idx + len(para)
                chunks.append(Chunk(
                    id=f"chunk_{i}",
                    content=para,
                    start_index=start_idx,
                    end_index=end_idx
                ))
                current_index = end_idx + 2
    else:
        # Default: paragraph-based chunking with size adjustment
        target_chunks = max(1, int(len(paragraphs) / chunk_size_factor))
        chunk_size = max(1, len(paragraphs) // target_chunks) if target_chunks > 0 else 1
        
        for chunk_idx in range(0, len(paragraphs), chunk_size):
            chunk_paras = paragraphs[chunk_idx:chunk_idx + chunk_size]
            chunk_text = "\n\n".join(chunk_paras)
            start_idx = current_index
            end_idx = start_idx + len(chunk_text)
            chunks.append(Chunk(
                id=f"chunk_{chunk_idx // chunk_size}",
                content=chunk_text,
                start_index=start_idx,
                end_index=end_idx
            ))
            current_index = end_idx + 2
    
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
@click.option("--use-peter-parser", is_flag=True, help="Use peter-parser for actual chunking")
@click.option("--document-type", type=click.Choice(["heading", "plain", "slide", "lifelog"]), help="Document type for peter-parser (4-case routing)")
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
    use_peter_parser: bool,
    document_type: Optional[str],
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
        try:
            from chunker_optimizer.infrastructure.llm.llm_client import LLMClient
        except ImportError:
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
    if use_peter_parser:
        if not PETER_PARSER_AVAILABLE:
            click.echo("❌ peter-parser is not available. Falling back to mock chunking.", err=True)
            chunking_function = create_chunking_function_from_context
        else:
            click.echo("🔧 Using peter-parser for chunking...")
            try:
                chunking_function = create_peter_parser_chunking_function(
                    chunk_unit=chunk_unit,
                    document_type=document_type,
                    prompt_content=initial_prompt
                )
                click.echo("✅ peter-parser chunking function created")
            except Exception as e:
                click.echo(f"⚠️  Failed to create peter-parser chunking function: {e}", err=True)
                click.echo("⚠️  Falling back to mock chunking", err=True)
                chunking_function = create_chunking_function_from_context
    else:
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
    click.echo(f"   - Using peter-parser: {use_peter_parser and PETER_PARSER_AVAILABLE}")
    if document_type:
        click.echo(f"   - Document type: {document_type}")
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
    
    # Save results (default to results folder if not specified)
    if not output:
        # Generate default filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"results/optimization_{timestamp}.json"
    
    output_path = Path(output)
    # Ensure results folder exists
    if output_path.parent.name == "results" or str(output_path).startswith("results/"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # If output doesn't start with results/, put it in results folder
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        output_path = results_dir / output_path.name
    
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
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    click.echo(f"\n💾 Results saved to: {output_path}")
    
    # Display chunking samples from final iteration
    if result.history:
        final_iteration = result.history[-1]
        if "chunk_samples" in final_iteration and final_iteration["chunk_samples"]:
            click.echo(f"\n📦 Chunking Results (showing {len(final_iteration['chunk_samples'])} of {final_iteration['chunk_count']} chunks):")
            click.echo("   " + "-" * 66)
            for i, chunk_sample in enumerate(final_iteration["chunk_samples"], 1):
                click.echo(f"\n   Chunk {i} (ID: {chunk_sample['id']}):")
                click.echo(f"   Length: {chunk_sample['length']} chars | Range: [{chunk_sample['start_index']}, {chunk_sample['end_index']})")
                preview = chunk_sample['content_preview']
                # Display preview with better formatting
                lines = preview.split('\n')[:3]  # Show first 3 lines
                for line in lines:
                    if line.strip():
                        click.echo(f"   {line[:60]}{'...' if len(line) > 60 else ''}")
                if len(preview) > 200:
                    click.echo("   ...")
            
            click.echo(f"\n   💡 전체 chunking 결과는 {output_path} 파일에서 확인할 수 있습니다.")
            click.echo(f"   💡 각 iteration별 chunk 샘플이 history에 저장되어 있습니다.")
    
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
