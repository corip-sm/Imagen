The initial fused Metal kernel patterns in `gaon_mflux_kernels.py` were adapted
from the FLUX Klein MLX runtime by devin-lai:
https://github.com/devin-lai/flux-3-image-klein-mlx

Its Apache License 2.0 is included in the root `LICENSE`.

The typed Sigmoid formula and RMS reduction order follow Apple MLX 0.32.4:
https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/unary_ops.h
https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/rms_norm.metal

Copyright © 2023 Apple Inc. MLX's MIT license is included in
`licenses/MLX-MIT.txt`.

Project changes preserve native LayerNorm, typed arithmetic and reduction order;
pack Q/K/V before evaluation; process packed Q/K/V directly; and reconstruct
existing Q8 weights for GPU BF16 and CPU FP32 execution. The on-disk checkpoint
is not rewritten. These changes activate only through the project launcher.
