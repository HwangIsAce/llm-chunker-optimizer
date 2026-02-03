# LLM Chunker Optimizer

LLM-based chunking optimizer with evaluation metrics for Retrieval-Augmented Generation (RAG) systems.

## Overview

This module provides an intelligent chunking optimization system that uses Large Language Models (LLMs) to iteratively improve text chunking prompts based on evaluation metrics. It implements metrics from research papers (MoC, HOPE) to assess chunk quality and automatically optimizes chunking strategies.

## Usage

### Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

### Configuration

1. Copy `env.example` to `.env`:
```bash
cp env.example .env
```

2. Set your OpenAI API key:
```env
OPENAI_API_KEY=your_api_key_here
```

### CLI

```bash
# Optimize chunking prompt
python -m chunker_optimizer.presentation.cli optimize \
    --document path/to/document.pdf \
    --initial-prompt "Split text into meaningful chunks" \
    --chunk-unit page \
    --threshold 0.8 \
    --max-iterations 10
```

### Programmatic Usage

```python
from chunker_optimizer.application.use_cases.run_optimization_loop import (
    RunOptimizationLoopUseCase,
    OptimizationConfig
)
from chunker_optimizer.domain.entities.prompt import Prompt
from chunker_optimizer.domain.entities.chunking_context import ChunkingContext

# Create use case
use_case = RunOptimizationLoopUseCase(...)

# Run optimization
config = OptimizationConfig(
    max_iterations=10,
    threshold=0.8
)

result = use_case.execute(
    initial_prompt=Prompt(...),
    chunking_context=ChunkingContext(...),
    config=config
)
```
