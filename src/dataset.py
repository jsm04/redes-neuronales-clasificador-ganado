"""
dataset.py — Carga, preprocesamiento y augmentation de datos.

Soporta múltiples tipos de problema:
  - image_classification: carga imágenes desde directorios por clase
  - text_sentiment: tokeniza y prepara texto para LSTM/Transformer
  - time_series: normaliza y crea ventanas temporales

Uso:
    python src/dataset.py --mode preprocess
"""

import os
import yaml
import argparse
from pathlib import Path

import numpy as np


# ============================================================
# CARGA DE CONFIGURACIÓN
# ============================================================

def load_config(config_path="config.yaml") -> dict:
    """Carga la configuración desde config.yaml."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# PREPROCESAMIENTO PARA CLASIFICACIÓN DE IMÁGENES
# ============================================================

def preprocess_image_dataset(config: dict):
    """
    Prepara dataset de imágenes.
    
    Espera estructura en data/raw/:
        data/raw/
            clase_0/
                img1.jpg
                img2.jpg
            clase_1/
                img1.jpg
                img3.png
    
    Guarda en data/preprocessed/:
        train.npy, val.npy, test.npy  (arrays de imágenes normalizadas)
        labels_train.npy, ...          (etiquetas correspondientes)
    """
    raw_dir = Path(config["data"]["raw_dir"])
    proc_dir = Path(config["data"]["preprocessed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Detectar clases a partir de subdirectorios
    classes = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])
    num_classes = len(classes)
    
    print(f"📂 Clases detectadas: {classes}")
    print(f"🔢 Número de clases: {num_classes}")

    # Cargar e imagenar datos
    X_train, y_train = load_images_from_dir(raw_dir / "clase_0", raw_dir / "clase_1")  # Ajustar según estructura real
    
    # Normalizar píxeles a [0, 1]
    X_train = X_train.astype("float32") / 255.0

    # Guardar en formato numpy
    np.save(proc_dir / "X_train.npy", X_train)
    np.save(proc_dir / "y_train.npy", y_train)
    
    print(f"✅ Dataset guardado en {proc_dir}")
    return num_classes


def load_images_from_dir(*class_dirs):
    """Carga imágenes desde directorios de clases."""
    from PIL import Image
    
    all_images = []
    all_labels = []
    
    for label, class_dir in enumerate(class_dirs):
        if not class_dir.exists():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                img = Image.open(img_path).convert("RGB")
                all_images.append(np.array(img))
                all_labels.append(label)
    
    return np.array(all_images), np.array(all_labels)


# ============================================================
# PREPROCESAMIENTO PARA ANÁLISIS DE TEXTO / SENTIMIENTOS
# ============================================================

def preprocess_text_dataset(config: dict):
    """
    Prepara dataset de texto para análisis de sentimientos.
    
    Espera un archivo CSV en data/raw/ con columnas: 'text', 'label'
    Guarda tokens y secuencias preparadas en data/preprocessed/.
    """
    import pandas as pd
    
    raw_dir = Path(config["data"]["raw_dir"])
    proc_dir = Path(config["data"]["preprocessed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Cargar dataset (ajustar nombre del archivo)
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        print("⚠️ No se encontraron CSV en data/raw/. Añade un archivo con 'text' y 'label'.")
        return 2  # Default binario

    df = pd.read_csv(csv_files[0])
    
    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else df.columns[-1]
    
    texts = df[text_col].astype(str).values
    labels = df[label_col].values
    
    # Tokenizar y secenciar (usando tokenizer de TF o sklearn)
    max_length = config["data"].get("text_preprocessing", {}).get("max_length", 512)
    
    # Simple approach: usar count vectorizer como placeholder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    
    vectorizer = TfidfVectorizer(max_features=config["data"]["text_preprocessing"].get("vocab_size", 10000))
    X = vectorizer.fit_transform(texts)
    
    # Normalizar labels a enteros si es necesario
    if labels.dtype == object:
        unique_labels = np.unique(labels)
        label_map = {l: i for i, l in enumerate(unique_labels)}
        y = np.array([label_map[l] for l in labels])
        num_classes = len(label_map)
    else:
        y = labels.astype(int)
        num_classes = len(np.unique(y))

    # Dividir dataset
    split_train = config["data"]["train_split"]
    split_val = config["data"]["validation_split"] / (1 - split_train)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X.toarray(), y, test_size=1 - split_train, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=split_val / (1 - split_train), 
        random_state=42, stratify=y_temp
    )

    # Guardar
    np.save(proc_dir / "X_train.npy", X_train.astype("float32"))
    np.save(proc_dir / "y_train.npy", y_train)
    np.save(proc_dir / "X_val.npy", X_val.astype("float32"))
    np.save(proc_dir / "y_val.npy", y_val)
    np.save(proc_dir / "X_test.npy", X_test.astype("float32"))
    np.save(proc_dir / "y_test.npy", y_test)
    
    print(f"✅ Dataset de texto guardado: {num_classes} clases")
    return num_classes


# ============================================================
# PREPROCESAMIENTO PARA SERIES TEMPORALES (placeholder)
# ============================================================

def preprocess_timeseries_dataset(config: dict):
    """
    Prepara dataset de series temporales.
    
    Espera archivo CSV con columnas de tiempo y valores numéricos.
    Crea ventanas deslizantes para predicción.
    """
    import pandas as pd
    
    raw_dir = Path(config["data"]["raw_dir"])
    proc_dir = Path(config["data"]["preprocessed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder: cargar CSV y normalizar valores entre 0 y 1
    print("⚙️ Preprocesamiento de series temporales (placeholder)")
    
    return 1  # Default


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Preprocesamiento de dataset")
    parser.add_argument("--mode", choices=["preprocess"], default="preprocess")
    args = parser.parse_args()
    
    config = load_config()
    
    problem_type = config["project"]["problem_type"]
    
    if "image" in problem_type:
        num_classes = preprocess_image_dataset(config)
    elif "text" in problem_type or "sentiment" in problem_type:
        num_classes = preprocess_text_dataset(config)
    else:
        num_classes = preprocess_timeseries_dataset(config)
    
    print(f"\n🏁 Preprocesamiento completado. {num_classes} clases detectadas.")


if __name__ == "__main__":
    main()
