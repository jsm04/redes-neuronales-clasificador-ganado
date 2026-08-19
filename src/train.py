"""
train.py — Loop principal de entrenamiento del modelo.

Carga datos preprocesados, construye el modelo según config.yaml,
entrena con callbacks (early stopping, checkpointing) y guarda resultados.

Uso:
    python src/train.py
    
O con Makefile:
    make train
"""

import os
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime

# Asegurar que el proyecto esté en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import load_config
from src.model import build_model


def train(config: dict = None):
    """Entrena el modelo completo según la configuración."""
    
    if config is None:
        config = load_config()
    
    print("=" * 60)
    print("🧠 INICIANDO ENTRENAMIENTO")
    print("=" * 60)
    print(f"   Tipo de problema: {config['project']['problem_type']}")
    print(f"   Arquitectura:     {config['model']['architecture']}")
    print(f"   Épocas:           {config['training']['epochs']}")
    print(f"   Batch size:       {config['training']['batch_size']}")
    print(f"   Learning rate:    {config['training']['learning_rate']}")
    print("=" * 60)
    
    # 1. Construir modelo
    model, num_classes = build_model(config)
    
    # 2. Cargar datos preprocesados
    proc_dir = Path(config["data"]["preprocessed_dir"])
    
    if "image" in config["project"]["problem_type"]:
        X_train = load_np_data(proc_dir / "X_train.npy")
        y_train = load_np_data(proc_dir / "y_train.npy")
    else:
        # Para texto/series temporales, cargar todos los splits
        X_train_raw = load_np_data(proc_dir / "X_train.npy")
        y_train = load_np_data(proc_dir / "y_train.npy")
        
        if (proc_dir / "X_val.npy").exists():
            X_val_raw = load_np_data(proc_dir / "X_val.npy")
            y_val = load_np_data(proc_dir / "y_val.npy")
        else:
            from sklearn.model_selection import train_test_split
            split_val = config["data"]["validation_split"]
            X_train_raw, X_val_raw, y_train, y_val = train_test_split(
                X_train_raw, y_train, test_size=split_val / (1 - config["data"]["train_split"]),
                random_state=42, stratify=y_train
            )
    
    # 3. Configurar callbacks
    callbacks = []
    
    # Early Stopping
    if config["training"]["early_stopping"]["enabled"]:
        es_config = config["training"]["early_stopping"]
        callbacks.append(
            __import__("tensorflow").keras.callbacks.EarlyStopping(
                monitor=es_config.get("monitor", "val_loss"),
                patience=es_config.get("patience", 10),
                restore_best_weights=True,
                verbose=1
            )
        )
    
    # Model Checkpoint (guardar mejor modelo)
    if config["training"]["checkpointing"]["enabled"]:
        filepath = Path(config["training"]["checkpointing"]["filepath"])
        filepath.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            __import__("tensorflow").keras.callbacks.ModelCheckpoint(
                str(filepath),
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            )
        )
    
    # 4. Entrenar
    print("\n🚀 Entrenando modelo...\n")
    
    history = model.fit(
        X_train_raw if "image" not in config["project"]["problem_type"] else X_train,
        y_train,
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        validation_data=(X_val_raw if "image" not in config["project"]["problem_type"] else None, 
                        y_val if "image" not in config["project"]["problem_type"] else None),
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Guardar métricas de entrenamiento
    save_metrics(history, config)
    
    print("\n✅ Entrenamiento completado!")
    return model, history


def load_np_data(filepath: Path):
    """Carga un archivo .npy."""
    import numpy as np
    if not filepath.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
    data = np.load(str(filepath))
    print(f"   📦 Cargado {str(filepath)}: shape={data.shape}, dtype={data.dtype}")
    return data


def save_metrics(history, config):
    """Guarda las métricas de entrenamiento en output/metrics.csv."""
    output_dir = Path(config["environment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import pandas as pd
    
    metrics_df = pd.DataFrame(history.history)
    metrics_path = output_dir / "training_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    
    # Guardar resumen en formato legible para el agente
    summary = {
        "timestamp": datetime.now().isoformat(),
        "epochs_run": len(history.history["loss"]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history.get("val_loss", ["N/A"])[-1]) if history.history.get("val_loss") else None,
        "final_accuracy": float(max(history.history["accuracy"])) if "accuracy" in history.history else None,
        "best_epoch": int(next(iter([i for i, v in enumerate(history.history.get("val_accuracy", [])) 
                          if v == max(history.history.get("val_accuracy", [0]))])))
    }
    
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Métricas guardadas en {metrics_path}")
    print(f"📋 Resumen guardado en {summary_path}")


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    train()


if __name__ == "__main__":
    main()
