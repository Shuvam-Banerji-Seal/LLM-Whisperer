# Generation Module

Generation and text augmentation for RAG systems.

## Overview

Assemble prompts and generate grounded responses:
- Prompt assembly from query and context
- Text augmentation strategies
- Generation orchestration
- Citation and confidence tracking
- Multiple generation modes (grounded, open-ended, etc.)

## Key Classes

### PromptAssembler
Constructs prompts from queries and context documents.

### TextAugmenter
Augments text with contextual information.

### GenerationOrchestrator
Orchestrates end-to-end generation pipeline.

## Usage

```python
from rag.generation import (
    PromptAssembler,
    GenerationOrchestrator,
    GenerationConfig,
    GenerationMode,
)

config = GenerationConfig(
    mode=GenerationMode.GROUNDED,
    llm_model="mistral-7b",
    include_citations=True,
)

assembler = PromptAssembler(config)
prompt = assembler.assemble(query, context_docs)

orchestrator = GenerationOrchestrator(
    retriever, reranker, assembler, config
)
result = orchestrator.generate(query, llm_generate_fn)
```

## Generation Modes

- **GROUNDED**: Response grounded in retrieved context
- **OPEN_ENDED**: Without strict context requirement
- **COMPARATIVE**: Compare across multiple documents
- **ABSTRACTIVE**: Summarize and synthesize

## Best Practices

1. Always include citations in production
2. Validate faithfulness to context
3. Use grounded mode for factual accuracy
4. Include confidence scores
5. Monitor hallucination rates

## References

- RAG main README: `../README.md`
- Configuration: `config.py`
- Core implementations: `core.py`
