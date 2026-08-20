# 🧠 Redes Neuronales — Proyecto Universitario

Sistema modular de redes neuronales para clasificación y predicción.

## Tipos de problema soportados

| Tipo | Datos | Arquitectura |
|------|-------|-------------|
| Clasificación de imágenes | directorios por clase en `data/raw/` | CNN / Simple CNN |

## Estructura

```
├── config.yaml           Configuración centralizada
├── requirements.txt      Dependencias Python
├── Makefile              Comandos automatizados
│
├── src/
│   ├── dataset.py        Carga y preprocesamiento de datos
│   ├── model.py          Arquitecturas (CNN, LSTM)
│   ├── train.py          Loop de entrenamiento
│   └── evaluate.py       Métricas y evaluación
│
├── tests/                Pruebas unitarias
├── data/raw/             Dataset original (colocar aquí)
├── models/               Modelos entrenados (.keras)
└── output/               Resultados generados
```

## Uso rápido

```bash
pip install -r requirements.txt   # instalar dependencias
# 1. Colocar dataset en data/raw/
#    - Imágenes: subcarpetas por clase (e.g. data/raw/clase_0/, clase_1/)
# 2. Configurar config.yaml (problem_type, arquitectura, hiperparámetros)
python src/dataset.py --mode preprocess
python src/train.py                  # entrenar modelo
python src/evaluate.py               # evaluar mejor modelo
pytest tests/                        # ejecutar tests
```

Ver `config.yaml` para configurar tipo de problema, arquitectura e hiperparámetros.

---

Proyecto académico universitario. Uso educativo.
