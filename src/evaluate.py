"""
evaluate.py — Evaluación del modelo entrenado.

Calcula métricas (accuracy, precision, recall, F1), genera matriz de confusión
y gráficos comparativos de entrenamiento. Guarda resultados en output/.

Uso:
    python src/evaluate.py --model models/best_model.keras
    
O con Makefile:
    make evaluate
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Asegurar que el proyecto esté en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import load_config


def evaluate(config: dict = None, model_path: str = None):
    """Evalúa un modelo entrenado y genera reportes."""
    
    if config is None:
        config = load_config()
    
    # Determinar path del modelo
    if model_path is None:
        ckpt_config = config["training"]["checkpointing"]
        model_path = str(Path(ckpt_config["filepath"]))
        
        if not Path(model_path).exists():
            print(f"⚠️ Modelo no encontrado en {model_path}")
            print("   Ejecuta primero: python src/train.py")
            return None
    
    print("=" * 60)
    print("📊 EVALUACIÓN DEL MODELO")
    print("=" * 60)
    print(f"   Modelo: {model_path}")
    print(f"   Problema: {config['project']['problem_type']}")
    
    # 1. Cargar modelo y datos
    import tensorflow as tf
    
    if not Path(model_path).exists():
        print("❌ Error: archivo del modelo no encontrado.")
        return None
        
    model = tf.keras.models.load_model(model_path)
    
    proc_dir = Path(config["data"]["preprocessed_dir"])
    
    # Cargar datos de test
    X_test_raw = np.load(str(proc_dir / "X_test.npy")) if (proc_dir / "X_test.npy").exists() else None
    y_test = np.load(str(proc_dir / "y_test.npy")) if (proc_dir / "y_test.npy").exists() else None
    
    # Si no hay test separado, usar validation set
    X_val_raw = np.load(str(proc_dir / "X_val.npy")) if (proc_dir / "X_val.npy").exists() else None
    y_val = np.load(str(proc_dir / "y_val.npy")) if (proc_dir / "y_val.npy").exists() else None
    
    test_X = X_test_raw if X_test_raw is not None else X_val_raw
    test_y = y_test if y_test is not None else y_val
    
    if test_X is None or test_y is None:
        print("⚠️ No se encontraron datos de prueba. Asegúrate de ejecutar preprocess primero.")
        return None
    
    # 2. Generar predicciones
    predictions = model.predict(test_X, verbose=0)
    
    if config["model"]["num_classes"] > 1:
        y_pred = np.argmax(predictions, axis=1)
    else:
        y_pred = (predictions > 0.5).astype(int).flatten()
    
    # 3. Calcular métricas
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    
    accuracy = accuracy_score(test_y, y_pred)
    
    print(f"\n   🎯 Accuracy: {accuracy:.4f}")
    print("\n" + classification_report(test_y, y_pred, digits=4))
    
    # 4. Matriz de confusión (si aplica)
    if config["evaluation"].get("confusion_matrix", True):
        cm = confusion_matrix(test_y, y_pred)
        print(f"\n📋 Matriz de Confusión:\n{cm}")
        
        # Guardar como CSV
        output_dir = Path(config["environment"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cm).to_csv(output_dir / "confusion_matrix.csv", index=False, header=False)
    
    # 5. Cargar métricas de entrenamiento si existen
    metrics_df_path = Path(config["environment"]["output_dir"]) / "training_metrics.csv"
    if metrics_df_path.exists():
        metrics_df = pd.read_csv(metrics_df_path)
        print(f"\n📈 Resumen de Entrenamiento:")
        for col in metrics_df.columns:
            last_val = metrics_df[col].iloc[-1]
            best_idx = metrics_df[col].idxmax() if "accuracy" in col else metrics_df[col].idxmin()
            best_val = metrics_df[col][best_idx]
            print(f"   {col}: último={last_val:.4f}, mejor={best_val:.4f} (epoch {best_idx+1})")
    
    # 6. Guardar reporte final
    output_dir = Path(config["environment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "accuracy": float(accuracy),
        "classification_report": str(classification_report(test_y, y_pred)),
        "confusion_matrix": cm.tolist() if config["evaluation"].get("confusion_matrix", True) else None
    }
    
    report_path = output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Reporte guardado en {report_path}")
    return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluación del modelo")
    parser.add_argument("--model", type=str, help="Ruta al modelo .keras/.h5/.pt")
    args = parser.parse_args()
    
    evaluate(model_path=args.model)


if __name__ == "__main__":
    main()
