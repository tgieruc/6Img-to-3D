import os
import tempfile

import pytest

from webui.backend.schemas.config_schema import FullConfig
from webui.backend.services.config_io import export_to_py, load_from_py

FIXTURE = "config/config.py"


def test_load_from_existing_config():
    if not os.path.exists(FIXTURE):
        pytest.skip("config/config.py not present")
    cfg = load_from_py(FIXTURE)
    assert isinstance(cfg, FullConfig)
    assert cfg.encoder.dim == 128
    assert cfg.encoder.tpv_h == 200
    assert cfg.encoder.num_encoder_layers == 5
    assert cfg.pif.enabled is True
    assert cfg.pif.factor == 0.125
    assert cfg.decoder.hidden_dim == 128
    assert cfg.optimizer.lr == pytest.approx(5e-5)
    assert cfg.dataset.train.batch_size == 1
    assert cfg.dataset.val.factor == pytest.approx(0.25)


def test_export_roundtrip():
    cfg = FullConfig()
    py_str = export_to_py(cfg)
    assert "_dim_" in py_str
    assert "N_h_" in py_str
    assert "TPVFormer" in py_str
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(py_str)
        tmp = f.name
    try:
        reloaded = load_from_py(tmp)
        assert reloaded.encoder.dim == cfg.encoder.dim
        assert reloaded.encoder.tpv_h == cfg.encoder.tpv_h
        assert reloaded.decoder.hidden_layers == cfg.decoder.hidden_layers
        assert reloaded.optimizer.lr == pytest.approx(cfg.optimizer.lr)
        assert reloaded.pif.enabled == cfg.pif.enabled
        assert reloaded.dataset.train.batch_size == cfg.dataset.train.batch_size
        assert reloaded.dataset.val.factor == pytest.approx(cfg.dataset.val.factor)
    finally:
        os.unlink(tmp)
