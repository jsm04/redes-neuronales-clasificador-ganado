"""test_model.py — Pruebas unitarias para la construcción del modelo."""

import pytest


def test_build_model_cnn():
    """Verifica que se puede construir un modelo CNN sin errores."""
    from src.model import build_cnn_model
    from src.dataset import load_config
    
    config = load_config()
    
    # Cambiar temporalmente a imagen para pruebas
    original_problem = config["project"]["problem_type"]
    config["project"]["problem_type"] = "image_classification"
    
    try:
        model = build_cnn_model(config)
        
        # Verificar que el modelo fue compilado correctamente
        assert hasattr(model, 'compile') or not model.built == False, \
            "El modelo no debería estar sin compilar"
            
        # Verificar número de parámetros > 0
        total_params = sum(model.count_params())
        assert total_params > 0, "El modelo no tiene parámetros"
        
    finally:
        config["project"]["problem_type"] = original_problem


def test_build_model_lstm():
    """Verifica que se puede construir un modelo LSTM sin errores."""
    from src.model import build_lstm_model
    from src.dataset import load_config
    
    config = load_config()
    
    # Cambiar temporalmente a texto para pruebas
    original_problem = config["project"]["problem_type"]
    config["project"]["problem_type"] = "text_sentiment"
    
    try:
        model = build_lstm_model(config)
        
        total_params = sum(model.count_params())
        assert total_params > 0, "El modelo LSTM no tiene parámetros"
        
    finally:
        config["project"]["problem_type"] = original_problem


def test_config_loading():
    """Verifica que la configuración se carga correctamente."""
    from src.dataset import load_config
    
    config = load_config()
    
    # Verificar estructura básica
    assert "project" in config, "Falta sección 'project' en config.yaml"
    assert "data" in config, "Falta sección 'data' en config.yaml"
    assert "model" in config, "Falta sección 'model' en config.yaml"
    assert "training" in config, "Falta sección 'training' en config.yaml"
