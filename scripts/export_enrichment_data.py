"""Script to export enrichment data from pipeline to JSON"""
import json
import click
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def export_enrichment_to_json(
    enrichment_data: Dict[str, Any],
    output_path: str,
    source_document: str = None,
    chunk_unit: str = "page"
) -> Path:
    """
    Export enrichment data to JSON file
    
    Args:
        enrichment_data: Pipeline extraction data
        output_path: Output file path
        source_document: Original document path
        chunk_unit: Chunk unit type
    
    Returns:
        Path to saved file
    """
    # Create JSON structure
    json_data = {
        "metadata": {
            "exported_at": datetime.now().isoformat(),
            "source_document": source_document or "unknown",
            "chunk_unit": chunk_unit,
            "version": "1.0"
        },
        "parsed_document": {
            "raw_content": enrichment_data.get("raw_content", ""),
            "parsed_elements": enrichment_data.get("parsed_elements", []),
            "metadata": enrichment_data.get("parse_metadata", {})
        },
        "vlm_outputs": [
            {
                "page_id": vlm.get("page_id", f"page_{i}"),
                "script": vlm.get("script", ""),
                "visual_elements": vlm.get("visual_elements", []),
                "metadata": vlm.get("metadata", {})
            }
            for i, vlm in enumerate(enrichment_data.get("vlm_outputs", []))
        ],
        "llm_summary": {
            "summary": enrichment_data.get("llm_summary", {}).get("summary", ""),
            "key_points": enrichment_data.get("llm_summary", {}).get("key_points", []),
            "topics": enrichment_data.get("llm_summary", {}).get("topics", []),
            "structure_info": enrichment_data.get("llm_summary", {}).get("structure_info", {}),
            "metadata": enrichment_data.get("llm_summary", {}).get("metadata", {})
        } if enrichment_data.get("llm_summary") else None,
        "enrichment_metadata": enrichment_data.get("enrichment_metadata", {})
    }
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    return output_file


@click.command()
@click.option("--document", required=True, help="Source document path")
@click.option("--output", help="Output file path (default: tests/fixtures/real_data/{chunk_unit}_enrichment.json)")
@click.option("--chunk-unit", default="page", help="Chunk unit: page, element, lifelog")
@click.option("--pipeline-module", default="your_langgraph_pipeline", help="Pipeline module path")
def export(document: str, output: str, chunk_unit: str, pipeline_module: str):
    """
    Export enrichment data from pipeline to JSON
    
    Usage example:
        python scripts/export_enrichment_data.py \\
            --document path/to/document.pdf \\
            --chunk-unit page \\
            --output tests/fixtures/real_data/page_enrichment_sample.json
    """
    click.echo(f"📄 Processing document: {document}")
    click.echo(f"📦 Chunk unit: {chunk_unit}")
    
    # Pipeline import and execution
    try:
        # TODO: Connect to actual pipeline
        # from your_langgraph_pipeline import run_pipeline
        # result = run_pipeline(document, chunk_unit=chunk_unit)
        
        click.echo("\n⚠️  Actual pipeline connection needed")
        click.echo("Please modify this script to:")
        click.echo("1. Import your actual pipeline module")
        click.echo("2. Call run_pipeline function")
        click.echo("3. Extract enrichment_data from result")
        
        # Example data structure
        enrichment_data = {
            "raw_content": "Sample document content...",
            "parsed_elements": [],
            "vlm_outputs": [],
            "llm_summary": None,
            "enrichment_metadata": {}
        }
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise
    
    # Determine output path
    if not output:
        output = f"tests/fixtures/real_data/{chunk_unit}_enrichment_sample.json"
    
    # Export to JSON
    output_file = export_enrichment_to_json(
        enrichment_data=enrichment_data,
        output_path=output,
        source_document=document,
        chunk_unit=chunk_unit
    )
    
    click.echo(f"\n✅ Exported to: {output_file}")
    click.echo(f"📊 Data summary:")
    click.echo(f"   - VLM outputs: {len(enrichment_data.get('vlm_outputs', []))}")
    click.echo(f"   - LLM summary: {'Yes' if enrichment_data.get('llm_summary') else 'No'}")


if __name__ == "__main__":
    export()
