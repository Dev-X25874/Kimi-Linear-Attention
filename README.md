# Kimi Linear Attention

Implementation of "Kimi Linear: An Expressive, Efficient Attention Architecture" (arXiv:2510.26692)

## Overview

Kimi Linear is a hybrid linear attention architecture that outperforms full attention across short-context, long-context, and RL scenarios. It combines Kimi Delta Attention (KDA) with Multi-Head Latent Attention (MLA) in a 3:1 ratio, achieving:

- **75% reduction in KV cache usage**
- **6x faster decoding** at 1M context length
- **Superior performance** vs. full attention with identical training

## Key Features

- **Kimi Delta Attention (KDA)**: Linear attention with fine-grained channel-wise gating
- **Hybrid Architecture**: 3 KDA layers + 1 MLA layer per block
- **No Position Encoding (NoPE)**: MLA layers delegate positional info to KDA
- **Chunkwise Parallel Training**: Efficient parallel processing
- **Hardware Optimized**: Specialized DPLR transition matrices

## Architecture

```
Input → Embedding → [KDA → KDA → KDA → MLA] × N → Output
                    └──────── Block ────────┘
```

**Key Design Choices:**
- KDA handles positional information via recurrent gating
- MLA provides global context without position embeddings
- 3:1 ratio balances efficiency and expressivity

## Installation

```bash
# Clone repository
git clone <repository-url>
cd kimi-linear-attention

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

### Training

```python
from kimi_linear import KimiLinearConfig, KimiLinearModel
import torch

# Configure model
config = KimiLinearConfig(
    vocab_size=50000,
    hidden_size=2048,
    num_layers=24,
    num_kda_layers=18,  # 3:1 ratio
    num_mla_layers=6,
    num_heads=16,
    kda_chunk_size=128,
)

# Initialize model
model = KimiLinearModel(config)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
input_ids = torch.randint(0, 50000, (2, 512))

outputs = model(input_ids, labels=input_ids)
loss = outputs.loss
loss.backward()
optimizer.step()
```

### Inference

```python
# Load pretrained model
model = KimiLinearModel.from_pretrained("checkpoint/")
model.eval()

# Generate
input_ids = torch.randint(0, 50000, (1, 10))
with torch.no_grad():
    output = model.generate(input_ids, max_length=100)
```

## Model Configurations

### Kimi-Linear-48B (Paper Configuration)
- Total Parameters: 48B
- Activated Parameters: 3B
- Layers: 60 (45 KDA + 15 MLA)
- Hidden Size: 5120
- Num Heads: 128

### Small (for testing)
- Total Parameters: ~1.5B
- Layers: 12 (9 KDA + 3 MLA)
- Hidden Size: 1024
- Num Heads: 16

## Performance

### vs Full Attention (MLA)

| Metric | Full MLA | Kimi Linear | Improvement |
|--------|----------|-------------|-------------|
| MMLU-Pro | - | - | +margin |
| RULER (128k) | - | 84.3 | Best |
| KV Cache | 100% | 25% | 75% reduction |
| Decode Speed (1M) | 1x | 6.3x | 6.3x faster |

## Project Structure

```
kimi-linear-attention/
├── kimi_linear/
│   ├── __init__.py
│   ├── config.py              # Model configuration
│   ├── kda.py                 # Kimi Delta Attention
│   ├── mla.py                 # Multi-Head Latent Attention
│   ├── model.py               # Full Kimi Linear model
│   ├── utils.py               # Helper functions
│   └── kernels/               # Optimized CUDA kernels
├── scripts/
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation
│   └── benchmark.py           # Benchmarking
├── tests/
│   ├── test_kda.py
│   ├── test_mla.py
│   └── test_model.py
├── configs/
│   ├── small.yaml
│   ├── base.yaml
│   └── large.yaml
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## Citation

```bibtex
@misc{team2025kimi,
  title={Kimi Linear: An Expressive, Efficient Attention Architecture},
  author={Zhang, Yu and Lin, Zongyu and Yao, Xingcheng and Hu, Jiaxi and others},
  journal={arXiv preprint arXiv:2510.26692},
  year={2025}
}
```

## License

MIT License

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

For bugs or issues, please open an issue on the repository.

## Acknowledgments

- Based on the paper by Moonshot AI team
- Built on PyTorch and Transformers
- Inspired by Gated DeltaNet and linear attention research

## Resources

- [Paper](https://arxiv.org/abs/2510.26692)
- [Official GitHub](https://github.com/MoonshotAI/Kimi-Linear)
- [Hugging Face Models](https://huggingface.co/moonshotai)
