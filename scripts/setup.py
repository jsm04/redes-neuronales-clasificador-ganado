#!/usr/bin/env python3
"""
setup.py — Script de configuración inicial del entorno.

Ejecuta una sola vez después de clonar el repo:
    python scripts/setup.py
    
Prepara:
  - Entorno virtual (opcional)
  - Instala dependencias
  - Crea estructura de directorios
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Ejecuta un comando y muestra resultado."""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Completado")
        else:
            print(f"   ⚠️ Error: {result.stderr}")
    except Exception as e:
        print(f"   ❌ Excepción: {e}")


def create_directories():
    """Crea la estructura de directorios."""
    base = Path(__file__).resolve().parent.parent
    
    dirs = [
        "data/raw",
        "data/preprocessed", 
        "models",
        "notebooks",
        "scripts",
        "tests",
        "output"
    ]
    
    for d in dirs:
        dir_path = base / d
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   📁 {dir_path}")


def install_dependencies():
    """Instala dependencias desde requirements.txt."""
    req_file = Path(__file__).resolve().parent.parent / "requirements.txt"
    
    if not req_file.exists():
        print("   ⚠️ requirements.txt no encontrado")
        return
    
    run_command(
        f"{sys.executable} -m pip install -r {req_file}",
        "Instalando dependencias"
    )


def create_virtual_env(use_venv: bool = True):
    """Crea entorno virtual si se desea."""
    if not use_venv:
        print("\n   ℹ️ Omitiendo creación de venv (usa tu propio entorno)")
        return
    
    base = Path(__file__).resolve().parent.parent
    venv_path = base / "venv"
    
    if venv_path.exists():
        print("   ℹ️ venv ya existe, omitiendo")
        return
    
    run_command(
        f"{sys.executable} -m venv {venv_path}",
        "Creando entorno virtual"
    )


def main():
    print("=" * 50)
    print("🧠 SETUP — Redes Neuronales")
    print("=" * 50)
    
    use_venv = input("\n¿Crear entorno virtual? [y/N]: ").strip().lower() == 'y'
    
    if use_venv:
        create_virtual_env(use_venv=True)
    
    create_directories()
    
    # Activar venv si se creó
    base = Path(__file__).resolve().parent.parent
    if (base / "venv" / "bin" / "activate").exists():
        print("\n   ℹ️ Activa el entorno con: source venv/bin/activate")
    
    install_dependencies()
    
    print("\n" + "=" * 50)
    print("✅ SETUP COMPLETADO")
    print("=" * 50)
    print("\nPróximos pasos:")
    print("  1. Define tu problema en config.yaml")
    print("  2. Descarga/coloca tus datos en data/raw/")
    print("  3. Ejecuta: make all")


if __name__ == "__main__":
    main()
