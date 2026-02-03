"""Script to export enrichment data from peter-parser pipeline to JSON"""
import json
import sys
import os
import click
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Add peter-parser to path if needed
PETER_PARSER_PATH = Path(__file__).parent.parent.parent.parent / "pipelines" / "parsing-pipeline" / "peter-parser"

# Load environment variables from .env files
# First try llm-chunker-optimizer's .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Then try peter-parser's .env (this takes precedence)
PETER_PARSER_ENV_PATH = PETER_PARSER_PATH / ".env"
if PETER_PARSER_ENV_PATH.exists():
    load_dotenv(PETER_PARSER_ENV_PATH, override=True)
if str(PETER_PARSER_PATH) not in sys.path:
    sys.path.insert(0, str(PETER_PARSER_PATH))

try:
    from peter_parser.graph.flow import PipelineFlow
    from peter_parser.graph.states import PipelineState, DocumentType
    from peter_parser_core.common.types import ParsedDocument, Chunk
except ImportError as e:
    click.echo(f"❌ Failed to import peter-parser: {e}", err=True)
    click.echo("Please ensure peter-parser is available in the workspace", err=True)
    sys.exit(1)


def convert_peter_parsed_doc_to_optimizer(
    parsed_doc: ParsedDocument
) -> Dict[str, Any]:
    """
    Convert peter-parser ParsedDocument to llm-chunker-optimizer format
    
    Args:
        parsed_doc: peter-parser ParsedDocument
        
    Returns:
        Dictionary in llm-chunker-optimizer ParsedDocument format
    """
    # Extract raw content
    raw_content = parsed_doc.content_text
    
    # Convert elements to parsed_elements format
    parsed_elements = []
    for element in parsed_doc.elements:
        parsed_elements.append({
            "element_id": element.element_id,
            "page_number": element.page_number,
            "category": element.category,
            "text": element.text,
            "coordinates": element.coordinates or [],
            "enrichment_metadata": element.enrichment_metadata or {},
            "chunk_uuid": element.chunk_uuid
        })
    
    # Extract metadata
    metadata = parsed_doc.metadata.copy() if parsed_doc.metadata else {}
    metadata["total_pages"] = len(parsed_doc.pages)
    metadata["total_elements"] = len(parsed_doc.elements)
    
    return {
        "raw_content": raw_content,
        "parsed_elements": parsed_elements,
        "metadata": metadata
    }


def convert_item_metadata_to_vlm_outputs(
    item_metadata: Dict[int, Dict[str, Any]],
    parsed_doc: ParsedDocument
) -> list[Dict[str, Any]]:
    """
    Convert peter-parser item_metadata to VLM outputs
    
    Args:
        item_metadata: Dict mapping element_id to metadata
        parsed_doc: ParsedDocument for reference
        
    Returns:
        List of VLM output dictionaries
    """
    vlm_outputs = []
    
    # Group by page_number
    page_metadata: Dict[int, Dict[str, Any]] = {}
    
    for element_id, metadata in item_metadata.items():
        # Find element to get page_number
        element = next(
            (e for e in parsed_doc.elements if e.element_id == element_id),
            None
        )
        
        if element is None:
            continue
            
        page_num = element.page_number
        
        if page_num not in page_metadata:
            page_metadata[page_num] = {
                "page_id": f"page_{page_num}",
                "script": metadata.get("vlm_script", metadata.get("script", "")),
                "visual_elements": metadata.get("visual_elements", []),
                "metadata": {
                    k: v for k, v in metadata.items()
                    if k not in ["vlm_script", "script", "visual_elements"]
                }
            }
        else:
            # Merge scripts if multiple elements on same page
            existing_script = page_metadata[page_num]["script"]
            new_script = metadata.get("vlm_script", metadata.get("script", ""))
            if new_script and new_script not in existing_script:
                page_metadata[page_num]["script"] = f"{existing_script}\n{new_script}"
            
            # Merge visual elements
            existing_elements = page_metadata[page_num]["visual_elements"]
            new_elements = metadata.get("visual_elements", [])
            page_metadata[page_num]["visual_elements"] = existing_elements + new_elements
    
    # Convert to list
    for page_num in sorted(page_metadata.keys()):
        vlm_outputs.append(page_metadata[page_num])
    
    return vlm_outputs


def convert_document_summary_to_llm_summary(
    document_summary: Optional[str],
    item_metadata: Optional[Dict[int, Dict[str, Any]]] = None
) -> Optional[Dict[str, Any]]:
    """
    Convert peter-parser document_summary to LLM summary format
    
    Args:
        document_summary: Document summary string
        item_metadata: Optional item metadata for additional info
        
    Returns:
        LLM summary dictionary or None
    """
    if not document_summary:
        return None
    
    # Try to extract structured info from summary or metadata
    key_points = []
    topics = []
    structure_info = {}
    
    # Simple extraction (can be enhanced)
    if item_metadata:
        # Try to find key points in metadata
        for metadata in item_metadata.values():
            if "key_points" in metadata:
                key_points.extend(metadata["key_points"])
            if "topics" in metadata:
                topics.extend(metadata["topics"])
    
    return {
        "summary": document_summary,
        "key_points": key_points if key_points else ["Document summary available"],
        "topics": topics if topics else [],
        "structure_info": structure_info,
        "metadata": {
            "source": "peter-parser",
            "extracted_at": datetime.now().isoformat()
        }
    }


def export_peter_parser_state_to_json(
    state: PipelineState,
    output_path: str,
    source_document: str = None,
    chunk_unit: str = "page"
) -> Path:
    """
    Export peter-parser PipelineState to llm-chunker-optimizer JSON format
    
    Args:
        state: PipelineState from peter-parser
        output_path: Output file path
        source_document: Original document path
        chunk_unit: Chunk unit type
        
    Returns:
        Path to saved file
    """
    parsed_doc = state.get("parsed_document")
    if not parsed_doc:
        raise ValueError("parsed_document is required in PipelineState")
    
    document_summary = state.get("document_summary")
    item_metadata = state.get("item_metadata", {})
    
    # Convert parsed document
    parsed_doc_data = convert_peter_parsed_doc_to_optimizer(parsed_doc)
    
    # Convert VLM outputs
    vlm_outputs = convert_item_metadata_to_vlm_outputs(item_metadata, parsed_doc)
    
    # Convert LLM summary
    llm_summary = convert_document_summary_to_llm_summary(document_summary, item_metadata)
    
    # Create JSON structure
    json_data = {
        "metadata": {
            "exported_at": datetime.now().isoformat(),
            "source_document": source_document or "unknown",
            "chunk_unit": chunk_unit,
            "version": "1.0",
            "source": "peter-parser"
        },
        "parsed_document": parsed_doc_data,
        "vlm_outputs": vlm_outputs,
        "llm_summary": llm_summary,
        "enrichment_metadata": {
            "processing_time": state.get("enrichment_metadata", {}).get("processing_time"),
            "chunk_unit": chunk_unit,
            "document_type": state.get("document_type"),
            "version": "1.0"
        }
    }
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    return output_file


@click.command()
@click.option("--document", required=True, help="Source document path (PDF file)")
@click.option("--output", help="Output file path (default: tests/fixtures/real_data/{chunk_unit}_enrichment.json)")
@click.option("--chunk-unit", default="page", type=click.Choice(["page", "element", "lifelog"]), help="Chunk unit")
@click.option("--document-type", type=click.Choice(["heading", "plain", "slide", "lifelog"]), help="Document type (4-case routing)")
def export(document: str, output: str, chunk_unit: str, document_type: Optional[str]):
    """
    Export enrichment data from peter-parser pipeline to JSON
    
    Usage example:
        python scripts/export_peter_parser_data.py \\
            --document path/to/document.pdf \\
            --chunk-unit page \\
            --output tests/fixtures/real_data/page_enrichment_sample.json
    """
    click.echo(f"📄 Processing document: {document}")
    click.echo(f"📦 Chunk unit: {chunk_unit}")
    if document_type:
        click.echo(f"📋 Document type: {document_type}")
    
    # Read document
    doc_path = Path(document)
    if not doc_path.exists():
        click.echo(f"❌ Document not found: {document}", err=True)
        sys.exit(1)
    
    try:
        with open(doc_path, "rb") as f:
            document_bytes = f.read()
    except Exception as e:
        click.echo(f"❌ Failed to read document: {e}", err=True)
        sys.exit(1)
    
    # Initialize pipeline
    try:
        click.echo("🔧 Initializing peter-parser pipeline...")
        pipeline = PipelineFlow()
    except Exception as e:
        click.echo(f"❌ Failed to initialize pipeline: {e}", err=True)
        click.echo("Please ensure peter-parser is properly configured", err=True)
        sys.exit(1)
    
    # Execute pipeline
    try:
        click.echo("🚀 Running pipeline...")
        doc_type: Optional[DocumentType] = document_type if document_type else None
        
        # Try to get partial results even if chunking fails
        try:
            state = pipeline.invoke(
                document=document_bytes,
                chunk_unit=chunk_unit if not document_type else None,
                document_type=doc_type
            )
            click.echo("✅ Pipeline execution completed")
        except Exception as chunk_error:
            # If chunking fails, try to get state from stream
            click.echo(f"⚠️  Chunking step failed: {chunk_error}", err=True)
            click.echo("🔄 Attempting to get partial results from stream...")
            
            # Try to get partial state from stream
            state = {}
            try:
                for chunk in pipeline.stream(
                    document=document_bytes,
                    chunk_unit=chunk_unit if not document_type else None,
                    document_type=doc_type
                ):
                    # Accumulate state from stream chunks
                    for key, value in chunk.items():
                        if key not in state:
                            state[key] = value
                        elif isinstance(state[key], list) and isinstance(value, list):
                            state[key].extend(value)
                        elif isinstance(state[key], dict) and isinstance(value, dict):
                            state[key].update(value)
                        else:
                            state[key] = value
                
                if not state:
                    raise chunk_error
                click.echo("✅ Partial results extracted from stream")
            except Exception as stream_error:
                click.echo(f"❌ Could not extract partial results: {stream_error}", err=True)
                raise chunk_error
    except Exception as e:
        click.echo(f"❌ Pipeline execution failed: {e}", err=True)
        import traceback
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)
    
    # Determine output path
    if not output:
        output = f"tests/fixtures/real_data/{chunk_unit}_enrichment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Export to JSON
    try:
        click.echo(f"💾 Exporting to JSON...")
        output_file = export_peter_parser_state_to_json(
            state=state,
            output_path=output,
            source_document=str(doc_path),
            chunk_unit=chunk_unit
        )
        
        click.echo(f"\n✅ Exported to: {output_file}")
        click.echo(f"📊 Data summary:")
        click.echo(f"   - Raw content length: {len(state.get('parsed_document', {}).content_text if hasattr(state.get('parsed_document'), 'content_text') else '')}")
        click.echo(f"   - Elements: {len(state.get('parsed_document', {}).elements if hasattr(state.get('parsed_document'), 'elements') else [])}")
        click.echo(f"   - VLM outputs: {len(state.get('item_metadata', {}))}")
        click.echo(f"   - LLM summary: {'Yes' if state.get('document_summary') else 'No'}")
        click.echo(f"   - Chunks: {len(state.get('chunks', []))}")
    except Exception as e:
        click.echo(f"❌ Export failed: {e}", err=True)
        import traceback
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


if __name__ == "__main__":
    export()
