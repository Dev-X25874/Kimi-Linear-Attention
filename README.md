# Kimi Linear Attention

PyTorch implementation of the architecture from "Kimi Linear: An Expressive, Efficient Attention Architecture" (arXiv 2510.26692).

## What this is

A hybrid attention architecture combining Kimi Delta Attention (KDA) with Multi-Head Latent Attention (MLA) in a 3 to 1 ratio, as described in the Moonshot AI paper. This repo implements the architecture in PyTorch. It has not been trained or benchmarked. No performance numbers in this README are from this implementation, only from the original paper.

## Features

Kimi Delta Attention: linear attention with channel-wise gating.
Hybrid blocks: 3 KDA layers plus 1 MLA layer.
No position encoding in MLA layers; positional info is handled by KDA.
Chunkwise parallel training.

## Installation

git clone https://github.com/Dev-X25874/Kimi-Linear-Attention.git
cd Kimi-Linear-Attention
pip install -r requirements.txt

## Usage

See examples in the scripts folder for model configuration and a basic forward and backward pass. No pretrained checkpoints are included.

## Citation

Zhang et al., Kimi Linear: An Expressive, Efficient Attention Architecture, arXiv 2510.26692, 2025.

## License

MIT

## Links

Paper: https://arxiv.org/abs/2510.26692
Official implementation: https://github.com/MoonshotAI/Kimi-Linear
