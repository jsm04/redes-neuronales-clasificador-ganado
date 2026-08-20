"""test_model.py — Verifica que las arquitecturas se construyen sin errores."""

import pytest


def test_build_cnn():
    """CNN se construye y compila correctamente."""
    from src.model import build_cnn_model
    from src.dataset import load_config
    
    config = load_config()
    model = build_cnn_model(config)
    
    assert sum(model.count_params()) > 0


def test_build_simple_cnn():
    """Simple CNN se construye y compila correctamente."""
    from src.model import build_simple_cnn
    from src.dataset import load_config
    
    config = load_config()
    model = build_simple_cnn(config)
    
    assert sum(model.count_params()) > 0


def test_build_lstm():
    """LSTM se construye y compila correctamente."""
    from src.model import build_lstm_model
    from src.dataset import load_config
    
    config = load_config()
    model = build_lstm_model(config)
    
    assert sum(model.count_params()) > 0
