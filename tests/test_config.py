"""test_config.py — Pruebas para la configuración del proyecto."""

import os
import pytest
from pathlib import Path


def test_project_structure():
    """Verifica que todos los directorios necesarios existen."""
    base_dir = Path(__file__).resolve().parent.parent
    
    required_dirs = [
        "src", "data/raw", "data/preprocessed", "models", 
        "notebooks", "scripts", "tests", "output"
    ]
    
    for dir_name in required_dirs:
        assert (base_dir / dir_name).exists(), f"Falta el directorio: {dir_name}"


def test_config_file_exists():
    """Verifica que config.yaml existe."""
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "config.yaml"
    
    assert config_path.exists(), "config.yaml no encontrado"


def test_requirements_file_exists():
    """Verifica que requirements.txt existe."""
    base_dir = Path(__file__).resolve().parent.parent
    req_path = base_dir / "requirements.txt"
    
    assert req_path.exists(), "requirements.txt no encontrado"


def test_source_files_exist():
    """Verifica que los archivos fuente principales existen."""
    base_dir = Path(__file__).resolve().parent.parent
    
    required_files = [
        "src/__init__.py",
        "src/dataset.py",
        "src/model.py",
        "src/train.py",
        "src/evaluate.py"
    ]
    
    for file_name in required_files:
        assert (base_dir / file_name).exists(), f"Falta el archivo: {file_name}"


def test_readme_exists():
    """Verifica que README.md existe."""
    base_dir = Path(__file__).resolve().parent.parent
    readme_path = base_dir / "README.md"
    
    assert readme_path.exists(), "README.md no encontrado"
