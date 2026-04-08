# RAG Tuning Module

## Overview

Fine-tuning module specialized for Retrieval-Augmented Generation (RAG) systems that combine retrieval and generation components.

## Key Features

- Joint retriever and generator training
- Retrieval loss and generation loss balance
- Dense retrieval support
- Top-k document retrieval

## Usage

```python
from fine_tuning.rag_tuning import RAGFinetuner, RAGTuningConfig

config = RAGTuningConfig(
    model_name="facebook/rag-token-base",
    retriever_model="facebook/dpr-question_encoder-single-nq-base",
    output_dir="./rag_output",
)

finetuner = RAGFinetuner(config)
finetuner.setup_model()
finetuner.setup_optimizer()
finetuner.setup_scheduler()
results = finetuner.train(train_loader, eval_loader)
```

## See Also

- [Base Module](../base/README.md)
- [LoRA Module](../lora/README.md)
