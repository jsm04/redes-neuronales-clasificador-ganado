# 🧠 Redes Neuronales — Proyecto Universitario

## Objetivo

Desarrollar un sistema de inteligencia artificial basado en redes neuronales que pueda resolver problemas de **clasificación** o **predicción**. El proyecto es modular y adaptable a distintos tipos de datos e arquitecturas.

### Posibles aplicaciones (elegir una)

| Tipo | Descripción | Arquitectura sugerida |
|------|-------------|----------------------|
| 🖼️ Clasificación de imágenes | Basura orgánica vs inorgánica | CNN (Convolutional Neural Network) |
| 📊 Predicción de series temporales | Precios de acciones, clima | LSTM / Transformer |
| 🎤 Reconocimiento de voz | Transcripción audio a texto | RNN + CTC Loss / Whisper-like |
| 💬 Generación / Análisis de texto | Chatbots, predicción de sentimientos | LSTM / BERT / GPT-small |

> **Estado actual:** Boilerplate genérico. Definir el objetivo y dataset antes de entrenar.

---

## Estructura del proyecto

```
redes-neuronales/
├── README.md              ← Estás aquí
├── AGENT.md               ← Documentación del agente AI autónomo
├── memoria.md             ← Memoria persistente del agente
├── skill.md               ← Habilidades y capacidades del agente
├── requirements.txt       ← Dependencias de Python
├── .gitignore            ← Archivos ignorados por Git
├── config.yaml           ← Configuración centralizada del proyecto
├── Makefile              ← Comandos automatizados para el agente
│
├── data/                 ← Datos del proyecto
│   ├── raw/              ← Dataset original descargado (no modificar)
│   └── preprocessed/     ← Datos limpios listos para entrenar
│
├── src/                  ← Código fuente principal
│   ├── dataset.py        ← Carga, limpieza y augmentation de datos
│   ├── model.py          ← Definición de arquitecturas (CNN, LSTM, etc.)
│   ├── train.py          ← Loop de entrenamiento
│   └── evaluate.py       ← Métricas y evaluación del modelo
│
├── models/               ← Modelos entrenados guardados (.h5, .pt)
├── notebooks/            ← Notebooks para experimentación (Google Colab / CodeLab)
├── scripts/              ← Scripts auxiliares (descarga, exportación)
├── tests/                ← Pruebas unitarias y de integración
│
└── output/               ← Resultados: gráficos, métricas, predicciones
```

---

## Requisitos

- Python 3.10+
- GPU recomendada (Google Colab / CodeLab para entrenamiento en la nube)
- Dataset específico (por definir según el objetivo elegido)

### Dependencias principales

| Paquete | Propósito |
|---------|-----------|
| `tensorflow` / `torch` | Framework de deep learning |
| `keras` | API de alto nivel (si usa TF) |
| `numpy`, `pandas` | Manipulación de datos |
| `matplotlib`, `seaborn` | Visualización |
| `scikit-learn` | Métricas de evaluación |
| `Pillow` | Procesamiento de imágenes (opcional) |
| `transformers` (HuggingFace) | Modelos de lenguaje pre-entrenados (opcional) |

---

## Cómo usar este boilerplate

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Preparar el dataset

Colocar los datos originales en `data/raw/`. El agente se encargará del preprocesamiento.

### 3. Configurar el proyecto

Editar `config.yaml` con las opciones del modelo, hiperparámetros y rutas.

### 4. Entrenar (vía script o notebook)

```bash
python src/train.py
# O abrir un notebook en notebooks/ para experimentación interactiva
```

### 5. Evaluar el modelo

```bash
python src/evaluate.py --model models/best_model.h5
```

---

## Agentes y automatización

Este proyecto está diseñado para ser dirigido por un **agente de IA autónomo** que puede:
- Ejecutar tareas de preprocesamiento, entrenamiento y evaluación sin intervención humana.
- Iterar sobre la arquitectura basándose en los resultados (feedback loop).
- Mantener una memoria del progreso y decisiones tomadas.

Ver `AGENT.md` y `skill.md` para más detalles.

---

## Licencia

Proyecto académico universitario. Uso educativo.
