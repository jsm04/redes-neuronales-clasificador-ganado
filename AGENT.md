# 🤖 AGENT.md — Agente de IA Autónoma para Redes Neuronales

## Identidad

Este es un **agente de automatización** encargado de gestionar el ciclo completo de desarrollo de modelos de redes neuronales dentro del proyecto `redes-neuronales`.

Su función principal es **ejecutar tareas de manera autónoma**: desde la descarga y preprocesamiento de datos hasta el entrenamiento, evaluación e iteración sobre la arquitectura.

---

## Capacidades principales

### 1. Preprocesamiento de datos
- Cargar datasets desde `data/raw/`
- Limpiar, normalizar y dividir en train/validation/test
- Aplicar augmentation (imágenes) o tokenización (texto)
- Guardar resultados en `data/preprocessed/`

### 2. Definición del modelo
- Seleccionar arquitectura según el tipo de problema:
  - **Imágenes** → CNN, ResNet, VGG, MobileNet
  - **Texto** → LSTM, GRU, Transformer, BERT
  - **Series temporales** → LSTM, Temporal Convolutional Network
- Configurar hiperparámetros desde `config.yaml`

### 3. Entrenamiento
- Ejecutar el loop de entrenamiento con monitoreo de métricas
- Guardar checkpoints intermedios en `models/`
- Registrar pérdida y precisión en `output/metrics.csv`

### 4. Evaluación
- Calcular métricas: accuracy, precision, recall, F1-score, loss
- Generar gráficos (curva de entrenamiento, matriz de confusión)
- Comparar resultados entre iteraciones

### 5. Feedback Loop (iteración inteligente)
Si la evaluación no pasa umbral mínimo:
- Analizar si el problema es **overfitting** o **underfitting**
- Ajustar arquitectura (más capas, dropout, learning rate)
- Registrar el cambio y re-entrenar automáticamente

---

## Flujo de trabajo autónomo

```
1. Cargar config.yaml → saber qué tipo de problema resolver
2. Descargar/preparar dataset si no existe en data/raw/
3. Preprocesar datos → data/preprocessed/
4. Definir arquitectura del modelo
5. Entrenar con validación cruzada
6. Evaluar métricas
7. ¿Métricas satisfactorias?
   ├─ SÍ → Guardar modelo final, generar reporte
   └─ NO → Analizar error, ajustar hiperparámetros, volver a paso 4
```

---

## Archivos clave que debe leer

| Archivo | Para qué |
|---------|----------|
| `config.yaml` | Configuración general del proyecto y modelo |
| `memoria.md` | Historial de decisiones y resultados previos |
| `src/dataset.py` | Lógica de carga y preprocesamiento |
| `src/model.py` | Definiciones de arquitecturas disponibles |
| `src/train.py` | Loop principal de entrenamiento |

---

## Convenciones del agente

- **Siempre** leer la configuración antes de ejecutar cualquier tarea.
- **Registrar** cada acción en `memoria.md`.
- **Guardar** modelos con nombre descriptivo: `modelo_<fecha>_<tipo>.h5` o `.pt`.
- **Documentar** cambios en hiperparámetros y resultados.
- Si un paso falla, intentar alternativas antes de detenerse.

---

## Toma de decisiones automática

El agente decide qué hacer basándose en:
1. **Tipo de problema** (definido en config.yaml)
2. **Estado actual** (leído de memoria.md)
3. **Resultados anteriores** (métricas históricas en memoria.md)
4. **Feedback de evaluaciones previas**

### Reglas de decisión:

```
SI accuracy < 0.7 → Probablemente underfitting
  → Aumentar capacidad del modelo
  
SI gap train/val > 0.2 → Overfitting detectado
  → Agregar dropout, data augmentation, o reducir modelo
  
SI loss no converge tras N epochs → Learning rate mal ajustado
  → Reducir learning rate a la mitad
```

## Comandos principales que puede ejecutar

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar preprocesamiento
python src/dataset.py --preprocess

# Entrenar modelo
python src/train.py

# Evaluar modelo
python src/evaluate.py --model models/best_model.h5

# Ver resultados
cat output/metrics.csv
```

O usar el Makefile:

```bash
make preprocess   # Preprocesamiento de datos
make train        # Entrenar modelo completo
make evaluate     # Evaluar último modelo
make all          # Pipeline completo: preprocess → train → evaluate
```
