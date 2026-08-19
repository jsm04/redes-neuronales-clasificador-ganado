# Script auxiliar — Descarga de datasets desde Kaggle
# Uso: python scripts/download_data.py --dataset "垃圾分类-dataset"

import argparse
import subprocess
from pathlib import Path


def download_from_kaggle(dataset_name: str, output_dir: str = "data/raw/"):
    """Descarga un dataset desde Kaggle."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Descargando dataset '{dataset_name}'...")
    
    # Instalar kaggle si no está
    subprocess.run(["pip", "install", "-q", "kaggle"], check=False)
    
    # Descargar (ajustar comando según fuente)
    cmd = f"kaggle datasets download -d {dataset_name} -p {output_path}"
    print(f"   Ejecutando: {cmd}")
    subprocess.run(cmd, shell=True)
    
    # Descomprimir si es ZIP
    import zipfile
    for zip_file in output_path.glob("*.zip"):
        print(f"   Descomprimiendo: {zip_file.name}")
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(output_path)
        zip_file.unlink()  # Eliminar ZIP después de extraer
    
    print(f"✅ Dataset descargado en {output_path}")


def download_from_huggingface(dataset_name: str):
    """Descarga un dataset desde HuggingFace (placeholder)."""
    print("⚙️ Descarga desde HuggingFace — requerido: pip install datasets")
    print(f"   Dataset: {dataset_name}")
    # from datasets import load_dataset
    # dataset = load_dataset(dataset_name)


def main():
    parser = argparse.ArgumentParser(description="Descargar datasets")
    parser.add_argument("--source", choices=["kaggle", "huggingface"], default="kaggle")
    parser.add_argument("--dataset", required=True, help="Nombre del dataset")
    args = parser.parse_args()
    
    if args.source == "kaggle":
        download_from_kaggle(args.dataset)
    else:
        download_from_huggingface(args.dataset)


if __name__ == "__main__":
    main()
