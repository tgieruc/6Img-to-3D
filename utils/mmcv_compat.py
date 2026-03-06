"""Compatibility shim for running the 6Img-to-3D encoder without a compiled
mmcv CUDA extension (mmcv._ext).

Import this module BEFORE anything from `triplane_encoder` or `builder`.
It patches mmcv so that:
  - ext_loader.load_ext() succeeds (returns a no-op stub)
  - mmcv.cnn gains constant_init / xavier_init / build_norm_layer from mmengine
  - mmcv.ops.multi_scale_deform_attn is importable

After importing this module and the encoder, call `force_pytorch_deform_attn()`
to patch the forward pass of TPVMSDeformableAttention3D to use the pure-PyTorch
fallback instead of the CUDA extension.

This is for TESTING ONLY. The production path (Docker / CI with nvcc) builds
mmcv from source and uses the CUDA kernel.
"""

import sys
import types
import warnings


def _patch_ext_loader():
    """Make ext_loader.load_ext() return a benign stub instead of crashing."""
    import mmcv.utils.ext_loader as _ext_loader

    def _load_ext_stub(name, funcs):
        """Return a NamedTuple-like stub whose attrs raise RuntimeError if called."""

        class _StubFn:
            def __init__(self, fn_name):
                self._fn_name = fn_name

            def __call__(self, *args, **kwargs):
                raise RuntimeError(
                    f"mmcv CUDA op '{self._fn_name}' was called but mmcv._ext is not "
                    "compiled. Install full mmcv (with CUDA) or use force_pytorch_deform_attn()."
                )

        stub_attrs = {fn: _StubFn(fn) for fn in funcs}
        Stub = types.SimpleNamespace(**stub_attrs)
        return Stub

    _ext_loader.load_ext = _load_ext_stub


def _patch_mmcv_cnn():
    """Add init helpers and build_norm_layer to mmcv.cnn from mmengine."""
    import mmcv.cnn as _cnn
    from mmcv.cnn.bricks.transformer import build_norm_layer
    from mmengine.model import constant_init, xavier_init

    if not hasattr(_cnn, "constant_init"):
        _cnn.constant_init = constant_init
    if not hasattr(_cnn, "xavier_init"):
        _cnn.xavier_init = xavier_init
    if not hasattr(_cnn, "build_norm_layer"):
        _cnn.build_norm_layer = build_norm_layer


def _patch_mmcv_ops():
    """Make mmcv.ops.multi_scale_deform_attn importable.

    Reads the function directly from the source file to bypass the
    mmcv.ops.__init__ that crashes when mmcv._ext is absent.
    """
    import importlib.util
    from pathlib import Path

    # Locate the file inside the installed mmcv package
    mmcv_path = Path(__import__("mmcv").__file__).parent
    attn_file = mmcv_path / "ops" / "multi_scale_deform_attn.py"

    if not attn_file.exists():
        warnings.warn(f"[mmcv_compat] Cannot find {attn_file}; pytorch deform-attn fallback unavailable")
        return

    # We need to patch ext_loader BEFORE importing the file's module-level code
    # ext_loader is already patched at this point, so the module-level
    # `ext_module = ext_loader.load_ext(...)` will use our stub.
    spec = importlib.util.spec_from_file_location("mmcv.ops.multi_scale_deform_attn", attn_file)
    mod = importlib.util.module_from_spec(spec)

    # Register first so inter-module imports resolve correctly
    sys.modules["mmcv.ops.multi_scale_deform_attn"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        warnings.warn(f"[mmcv_compat] Could not load multi_scale_deform_attn: {e}")
        # Provide a minimal module anyway

        def _pytorch_fallback(value, value_spatial_shapes, sampling_locations, attention_weights):
            raise RuntimeError("multi_scale_deformable_attn_pytorch not loaded")

        mod.multi_scale_deformable_attn_pytorch = _pytorch_fallback


def apply_patches():
    """Apply all mmcv compatibility patches. Call BEFORE importing triplane_encoder.

    Only stubs out the CUDA extension loader when mmcv._ext is not compiled.
    Always patches mmcv.cnn with constant_init/xavier_init (moved to mmengine in v2).
    """
    import mmcv  # noqa: F401 – trigger mmcv's own __init__ first

    try:
        import mmcv._ext  # noqa: F401

        _ext_compiled = True
    except ImportError:
        _ext_compiled = False

    if not _ext_compiled:
        _patch_ext_loader()
        _patch_mmcv_ops()

    _patch_mmcv_cnn()


def force_pytorch_deform_attn():
    """Patch TPVMSDeformableAttention3D to always use the PyTorch fallback.

    Call this AFTER importing triplane_encoder.  Without the mmcv CUDA
    extension, calling the CUDA path would raise RuntimeError.
    """
    try:
        from triplane_encoder.modules.image_cross_attention import TPVMSDeformableAttention3D

        _original_forward = TPVMSDeformableAttention3D.forward

        def _pytorch_forward(self, query, key, value, *args, **kwargs):
            # Run forward up to the branch; redirect CUDA path to pytorch
            import torch

            # Call the original but intercept the CUDA branch by temporarily
            # monkeypatching torch.cuda.is_available inside the method.
            _real_is_avail = torch.cuda.is_available

            def _fake_cuda_unavail():
                return False

            torch.cuda.is_available = _fake_cuda_unavail
            try:
                result = _original_forward(self, query, key, value, *args, **kwargs)
            finally:
                torch.cuda.is_available = _real_is_avail
            return result

        TPVMSDeformableAttention3D.forward = _pytorch_forward
    except Exception as e:
        warnings.warn(f"[mmcv_compat] force_pytorch_deform_attn failed: {e}")
