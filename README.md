# Playground

In this repo, I store a bunch of misc. scripts and notebooks that aren't necessarily associated with a larger project but allow me to test out different tools.

## Speech command classification

The speech command notebook uses the same CUDA 12.6 PyTorch build as
`rcw-graph-rag`. To avoid installing a second copy of the large CUDA runtime,
select the existing `rcw-graph-rag/.venv` Python interpreter as the notebook
kernel in VS Code. It already has the matching GPU-enabled PyTorch and
`ipykernel`.

The notebook reads the included Speech Commands dataset with Python's built-in
WAV reader and computes its mel spectrograms with PyTorch, so it does not need
`torchaudio` or TorchCodec.
