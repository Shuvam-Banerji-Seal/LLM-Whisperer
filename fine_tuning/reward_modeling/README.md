# Reward Modeling Module

## Overview

Training module for reward models used in Reinforcement Learning from Human Feedback (RLHF) and similar alignment techniques.

## Key Features

- Pairwise ranking loss
- Score normalization
- Margin-based training
- Temperature scaling

## Usage

```python
from fine_tuning.reward_modeling import RewardModelFinetuner, RewardModelingConfig

config = RewardModelingConfig(
    model_name="microsoft/phi-1.5",
    output_dir="./reward_model_output",
    use_pairwise_loss=True,
    margin=1.0,
)

finetuner = RewardModelFinetuner(config)
finetuner.setup_model()
finetuner.setup_optimizer()
finetuner.setup_scheduler()
results = finetuner.train(train_loader, eval_loader)
```

## See Also

- [Base Module](../base/README.md)
