"""Enable the project FLUX execution path for invocations from generate.sh."""

import os


if os.environ.get("GAON_MFLUX_FUSED_KERNELS") == "1":
    from gaon_mflux_kernels import enable as enable_kernels
    from gaon_mflux_runtime import enable as enable_runtime

    enable_kernels()
    enable_runtime()
