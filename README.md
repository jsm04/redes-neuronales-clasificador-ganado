# 🧠 Redes Neuronales — Proyecto Universitario

Proyecto de redes neuronales para **clasificación de imágenes de ganado** utilizando Python, TensorFlow/Keras y `uv` para la gestión del entorno y las dependencias.

El objetivo inicial del proyecto es entrenar una red neuronal convolucional (CNN) capaz de clasificar imágenes en diferentes clases. La configuración del proyecto se encuentra centralizada en `config.yaml`.

## 1. Requisitos

Se necesita:

* Python 3.11 o superior
* `uv`
* GPU NVIDIA compatible con CUDA, opcional pero recomendada para entrenamiento
* Git

El proyecto utiliza `uv` para:

* crear el entorno virtual;
* instalar las dependencias;
* ejecutar scripts;
* ejecutar tests.

No es necesario utilizar `pip`, `venv`, `requirements.txt` ni instalar las dependencias globalmente.

---

## 2. Instalar `uv`

### Linux / macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Después de instalarlo, reiniciar la terminal o cargar nuevamente el shell.

Comprobar:

```bash
uv --version
```

### Windows

Desde PowerShell:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Comprobar:

```powershell
uv --version
```

---

## 3. Obtener el proyecto

Clonar el repositorio:

```bash
git clone <URL_DEL_REPOSITORIO>
cd redes-neuronales
```

---

## 4. Crear el entorno e instalar dependencias

Crear el entorno virtual:

```bash
uv venv
```

Activar el entorno es opcional cuando se utiliza `uv run`, pero puede hacerse manualmente.

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

Instalar las dependencias del proyecto:

```bash
uv sync
```

Esto crea/utiliza el entorno `.venv` y sincroniza las dependencias definidas en `pyproject.toml` y `uv.lock`.

Para comprobar que Python funciona:

```bash
uv run python --version
```

También se puede comprobar TensorFlow:

```bash
uv run python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## 5. Estructura del proyecto

```text
.
├── config.yaml
├── pyproject.toml
├── uv.lock
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── tests/
│
├── data/
│   └── raw/
│       ├── cattle/
│       └── not_cattle/
│
└── output/
    ├── best_model.keras
    ├── final_model.keras
    ├── class_names.json
    ├── training_metrics.csv
    ├── training_summary.json
    └── ...
```

### Descripción

| Archivo/directorio | Función                                   |
| ------------------ | ----------------------------------------- |
| `config.yaml`      | Configuración central del proyecto        |
| `pyproject.toml`   | Dependencias y configuración de Python    |
| `uv.lock`          | Versiones exactas de las dependencias     |
| `src/dataset.py`   | Carga, división y preparación del dataset |
| `src/model.py`     | Definición de la arquitectura CNN         |
| `src/train.py`     | Entrenamiento del modelo                  |
| `src/evaluate.py`  | Evaluación y generación de métricas       |
| `tests/`           | Pruebas automatizadas                     |
| `data/raw/`        | Dataset original                          |
| `output/`          | Modelos, métricas y resultados generados  |

---

# 6. Preparar el dataset

El proyecto utiliza clasificación de imágenes mediante directorios.

Para una clasificación binaria `cattle` / `not_cattle`, la estructura debe ser:

```text
data/
└── raw/
    ├── cattle/
    │   ├── image001.jpg
    │   ├── image002.jpg
    │   ├── image003.jpg
    │   └── ...
    │
    └── not_cattle/
        ├── image001.jpg
        ├── image002.jpg
        ├── image003.jpg
        └── ...
```

Cada subdirectorio representa una clase.

Por ejemplo:

```text
data/raw/cattle/
```

contiene imágenes que pertenecen a la clase `cattle`, mientras que:

```text
data/raw/not_cattle/
```

contiene imágenes que no pertenecen a esa clase.

No es necesario renombrar las imágenes ni crear archivos `.npy`.

El dataset se carga directamente desde disco utilizando `tf.data`.

---

# 7. Configurar el proyecto

La configuración principal se encuentra en:

```text
config.yaml
```

La configuración actual está orientada a clasificación de imágenes:

```yaml
project:
  name: "redes-neuronales"
  problem_type: "image_classification"
```

El dataset se configura mediante:

```yaml
data:
  raw_dir: "data/raw/"
  batch_size: 32
  train_split: 0.70
  validation_split: 0.15
  test_split: 0.15
  random_seed: 42
```

La división utilizada será:

```text
70% → entrenamiento
15% → validación
15% → test
```

La arquitectura se configura mediante:

```yaml
model:
  architecture: "cnn"
  input_shape: [224, 224, 3]
  num_classes: 2
```

`num_classes` corresponde al número de directorios/clases encontrados en el dataset y puede ser actualizado automáticamente por el código.

Los principales hiperparámetros de entrenamiento son:

```yaml
training:
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam"
```

---

# 8. Inspeccionar el dataset

Antes de entrenar, se puede ejecutar:

```bash
uv run python src/dataset.py
```

Esto permite comprobar que el dataset se encuentra correctamente estructurado y que las clases pueden ser detectadas.

La salida debería mostrar información equivalente a:

```text
Clases encontradas:
- cattle
- not_cattle

Total de imágenes: ...
```

Si las clases no aparecen correctamente, revisar la estructura de:

```text
data/raw/
```

---

# 9. Entrenar el modelo

El entrenamiento se ejecuta mediante:

```bash
uv run python src/train.py
```

El script:

1. carga `config.yaml`;
2. detecta las clases;
3. divide el dataset en entrenamiento, validación y test;
4. crea los datasets de TensorFlow;
5. aplica data augmentation únicamente al conjunto de entrenamiento;
6. construye la CNN;
7. entrena el modelo;
8. utiliza Early Stopping;
9. guarda el mejor modelo;
10. guarda el modelo final;
11. guarda las métricas del entrenamiento.

Los resultados se almacenan en:

```text
output/
```

Por ejemplo:

```text
output/
├── best_model.keras
├── final_model.keras
├── class_names.json
├── training_metrics.csv
└── training_summary.json
```

---

# 10. Evaluar el modelo

Después del entrenamiento:

```bash
uv run python src/evaluate.py
```

Por defecto, el evaluador utiliza el modelo configurado en el proyecto o el modelo final disponible.

También se puede indicar explícitamente un modelo:

```bash
uv run python src/evaluate.py --model output/best_model.keras
```

La evaluación utiliza exclusivamente el conjunto de **test**.

No se debe utilizar el conjunto de entrenamiento para evaluar el rendimiento final del modelo.

Se generan métricas como:

* accuracy;
* precision;
* recall;
* F1-score;
* métricas por clase;
* matriz de confusión;
* loss;
* evolución de las métricas durante el entrenamiento.

Los resultados se almacenan en:

```text
output/
```

---

# 11. Ejecutar los tests

Para ejecutar todas las pruebas:

```bash
uv run pytest
```

También se puede utilizar:

```bash
uv run pytest tests/
```

Para obtener mayor detalle:

```bash
uv run pytest -v
```

---

# 12. Flujo completo desde cero

Después de clonar el proyecto, el flujo completo es:

```bash
# 1. Entrar al proyecto
cd redes-neuronales

# 2. Crear el entorno
uv venv

# 3. Instalar dependencias
uv sync

# 4. Colocar el dataset
# data/raw/cattle/
# data/raw/not_cattle/

# 5. Verificar el dataset
uv run python src/dataset.py

# 6. Entrenar
uv run python src/train.py

# 7. Evaluar
uv run python src/evaluate.py

# 8. Ejecutar tests
uv run pytest
```

No es necesario activar `.venv` para ninguno de estos comandos porque `uv run` ejecuta los comandos dentro del entorno administrado por `uv`.

---

# 13. Desarrollo y gestión de dependencias con `uv`

Para agregar una dependencia:

```bash
uv add nombre-paquete
```

Por ejemplo:

```bash
uv add tensorflow
```

Para agregar una dependencia únicamente de desarrollo:

```bash
uv add --dev pytest
```

Para eliminar una dependencia:

```bash
uv remove nombre-paquete
```

Después de modificar las dependencias, `uv` actualiza:

```text
pyproject.toml
uv.lock
```

Para sincronizar nuevamente el entorno:

```bash
uv sync
```

Para actualizar las dependencias:

```bash
uv lock --upgrade
uv sync
```

---

# 14. Ejecutar comandos Python

No es necesario utilizar:

```bash
python ...
```

Preferentemente utilizar:

```bash
uv run python ...
```

Por ejemplo:

```bash
uv run python src/train.py
```

Para ejecutar un módulo:

```bash
uv run python -m pytest
```

Para abrir un intérprete Python:

```bash
uv run python
```

---

# 15. GPU NVIDIA

El entrenamiento puede utilizar una GPU NVIDIA si el entorno de TensorFlow está correctamente configurado.

Para comprobar si TensorFlow detecta una GPU:

```bash
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Una salida con una GPU indica que TensorFlow puede acceder al dispositivo.

También se puede comprobar desde el sistema:

```bash
nvidia-smi
```

Si TensorFlow no detecta la GPU, el entrenamiento puede ejecutarse en CPU, aunque será considerablemente más lento.

La configuración específica de CUDA/cuDNN depende del sistema operativo y de la versión de TensorFlow instalada, por lo que debe verificarse contra las versiones utilizadas por el proyecto.

---

# 16. Archivos generados

Los resultados generados durante entrenamiento y evaluación se almacenan en:

```text
output/
```

Estos archivos no forman parte del código fuente.

Ejemplos:

```text
output/
├── best_model.keras
├── final_model.keras
├── class_names.json
├── training_metrics.csv
├── training_summary.json
├── confusion_matrix.csv
├── training_history.png
└── evaluation_*.json
```

Los modelos entrenados y resultados experimentales no deberían almacenarse dentro de `src/`.

---

# 17. Reproducibilidad

El proyecto utiliza una semilla configurable:

```yaml
data:
  random_seed: 42
```

Esto permite mantener una división reproducible del dataset entre entrenamiento, validación y test.

La reproducibilidad completa de un entrenamiento de redes neuronales puede depender además de:

* versión de Python;
* versión de TensorFlow;
* versión de CUDA;
* versión del driver NVIDIA;
* hardware;
* operaciones paralelas;
* configuración de GPU.

Por este motivo, `pyproject.toml` y `uv.lock` forman parte del proyecto y deben mantenerse versionados.

---

# 18. Buenas prácticas

No modificar manualmente los archivos generados dentro de `output/`.

No colocar imágenes directamente en:

```text
data/raw/
```

Las imágenes deben estar dentro de un directorio que represente su clase.

Correcto:

```text
data/raw/cattle/image001.jpg
```

Incorrecto:

```text
data/raw/image001.jpg
```

No utilizar las imágenes del conjunto de test durante el entrenamiento.

El conjunto de test debe reservarse para la evaluación final del modelo.

Las modificaciones de hiperparámetros deben realizarse en:

```text
config.yaml
```

y no directamente dentro de `train.py` o `model.py`, siempre que el parámetro esté contemplado por la configuración.

---

## 19. Objetivo del proyecto

El primer objetivo es desarrollar un clasificador CNN capaz de distinguir correctamente entre imágenes de ganado y otras imágenes.

Una vez validado este pipeline, el proyecto puede extenderse hacia problemas más complejos, como:

* detección de ganado mediante bounding boxes;
* conteo de animales;
* clasificación de razas;
* seguimiento de animales en vídeo;
* identificación individual de animales.

Estas tareas requieren arquitecturas y datasets diferentes al problema inicial de clasificación de imágenes.

---

## 19. Backend y frontend

El entrenamiento vive en `colab/py/red_neuronal_ganado.py` y no depende del backend. La API solo carga el checkpoint ya entrenado.

### API de inferencia

```bash
uv pip install --python .venv/bin/python -e backend
.venv/bin/uvicorn src.main:app --reload --port 8000
```

Endpoints:

* `GET /health` — estado, dispositivo y clases del checkpoint.
* `POST /api/v1/predictions` — campo `file` con la imagen (multipart).

Variables de entorno opcionales: `CATTLE_CHECKPOINT` (ruta al `.pth`),
`CATTLE_DEVICE` (`cuda`/`cpu`), `CATTLE_TOP_K`, `CATTLE_CORS_ORIGINS`,
`CATTLE_MAX_UPLOAD_BYTES`.

### Interfaz web

```bash
cd frontend
bun install   # o npm install
bun run dev   # http://localhost:3000
```

`frontend/.env.local` define `NEXT_PUBLIC_API_URL` (por defecto
`http://localhost:8000`).

### Tests

```bash
.venv/bin/python -m pytest tests -q
```

---

Proyecto académico universitario. Uso educativo.
