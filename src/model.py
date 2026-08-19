"""
model.py — Definición de arquitecturas de redes neuronales.

Arquitecturas disponibles:
  - cnn            → Clasificación de imágenes (CNN estándar)
  - simple_cnn     → CNN ligera para datasets pequeños
  - lstm           → Secuencias de texto / series temporales
  - transformer    → Attention-based model (placeholder)
  
Uso:
    from src.model import build_model
    
    config = load_config()
    model, num_classes = build_model(config)
"""

import yaml
from pathlib import Path


def load_config(config_path="config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# CNN — Clasificación de Imágenes
# ============================================================

def build_cnn_model(config: dict):
    """Construye una CNN estándar para clasificación de imágenes."""
    import tensorflow as tf
    from tensorflow.keras import layers, models
    
    cnn_config = config["model"].get("cnn", {})
    reg_config = config["model"].get("regularization", {})
    
    dropout_rate = reg_config.get("dropout_rate", 0.5)
    l2_reg = reg_config.get("l2_regularization", 0.01)
    
    model = models.Sequential([
        # Bloque 1
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=config["model"]["input_shape"],
                      kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Bloque 2
        layers.Conv2D(64, (3, 3), activation="relu",
                      kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Bloque 3
        layers.Conv2D(128, (3, 3), activation="relu",
                      kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Clasificador
        layers.Flatten(),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation="relu"),
        layers.Dropout(dropout_rate * 0.5),
        layers.Dense(config["model"]["num_classes"], activation="softmax")
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


# ============================================================
# SIMPLE CNN — Para datasets pequeños o prototipado rápido
# ============================================================

def build_simple_cnn(config: dict):
    """CNN ligera para datasets pequeños."""
    import tensorflow as tf
    
    reg_config = config["model"].get("regularization", {})
    dropout_rate = reg_config.get("dropout_rate", 0.3)
    
    model = tf.keras.Sequential([
        layers.Conv2D(16, (3, 3), activation="relu", input_shape=config["model"]["input_shape"]),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation="relu"),
        layers.Dense(config["model"]["num_classes"], activation="softmax")
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


# ============================================================
# LSTM — Secuencias de Texto / Series Temporales
# ============================================================

def build_lstm_model(config: dict):
    """LSTM para análisis de texto o series temporales."""
    import tensorflow as tf
    
    lstm_config = config["model"].get("lstm", {})
    reg_config = config["model"].get("regularization", {})
    
    model = tf.keras.Sequential([
        layers.Embedding(
            input_dim=config["data"]["text_preprocessing"].get("vocab_size", 10000),
            output_dim=64,
            input_length=config["data"]["text_preprocessing"].get("max_length", 512)
        ),
        layers.LSTM(lstm_config.get("units", 128)),
        layers.Dropout(reg_config.get("dropout_rate", 0.3)),
        layers.Dense(64, activation="relu"),
        layers.Dense(config["model"]["num_classes"], activation="softmax")
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


# ============================================================
# TRANSFORMER — Placeholder para modelos attention-based
# ============================================================

def build_transformer_model(config: dict):
    """Placeholder para modelo Transformer (requiere más desarrollo)."""
    print("⚙️ Modelo Transformer no está implementado aún.")
    print("   Usar pre-trained BERT de HuggingFace como alternativa.")
    return None


# ============================================================
# FACTORY — Selección automática según configuración
# ============================================================

def build_model(config: dict = None):
    """
    Fábrica que construye el modelo correcto según config.yaml.
    
    Args:
        config: Diccionario de configuración (si None, carga desde archivo)
        
    Returns:
        model: Modelo compilado listo para entrenar
        num_classes: Número de clases para la capa final
    
    Raises:
        ValueError: Si la arquitectura no está soportada
    """
    if config is None:
        config = load_config()
    
    arch = config["model"]["architecture"]
    print(f"🏗️ Construyendo modelo con arquitectura: {arch}")
    
    builders = {
        "cnn": build_cnn_model,
        "simple_cnn": build_simple_cnn,
        "lstm": build_lstm_model,
        "transformer": build_transformer_model,
    }
    
    if arch not in builders:
        raise ValueError(f"Arquitectura '{arch}' no soportada. Opciones: {list(builders.keys())}")
    
    model = builders[arch](config)
    num_classes = config["model"]["num_classes"]
    
    print(f"✅ Modelo construido: {sum(model.count_params()):,} parámetros totales")
    return model, num_classes


# ============================================================
# ENTRY POINT (para ejecutar desde línea de comandos)
# ============================================================

def main():
    config = load_config()
    model, num_classes = build_model(config)
    
    model.summary()
    
    # Guardar arquitectura como JSON
    json_path = Path("output") / "model_architecture.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        f.write(model.to_json())
    print(f"\n📁 Arquitectura guardada en {json_path}")


if __name__ == "__main__":
    main()
