# 🎯 Skill — Habilidades del Agente Autónomo

## Visión general

El agente opera como un **científico de datos automatizado** especializado en redes neuronales. Puede ejecutar tareas completas sin supervisión humana, desde la preparación de datos hasta el ajuste fino del modelo.

---

## Habilidades principales

### 1. 📊 Gestión de datos
- **Descargar datasets** desde fuentes públicas (Kaggle, UCI, HuggingFace)
- **Preprocesar datos:** limpieza, normalización, división train/val/test
- **Data augmentation:** rotación, zoom, flip (imágenes); tokenización, padding (texto)
- **Validar integridad** del dataset antes de entrenar

### 2. 🏗️ Diseño de arquitectura
- Seleccionar entre CNN, LSTM, Transformer según el tipo de dato
- Configurar número de capas, neuronas, funciones de activación
- Aplicar regularización (dropout, L2) cuando detecte overfitting
- Optimizar hyperparámetros (learning rate, batch size, epochs)

### 3. 🔄 Entrenamiento y monitoreo
- Ejecutar loops de entrenamiento con callbacks (early stopping, checkpointing)
- Monitorear loss y accuracy en tiempo real
- Guardar los mejores modelos según validación
- Visualizar curvas de entrenamiento

### 4. 📈 Evaluación automatizada
- Calcular métricas: Accuracy, Precision, Recall, F1, AUC-ROC
- Generar matriz de confusión y gráficas comparativas
- Comparar resultados entre diferentes iteraciones
- Detectar overfitting vs underfitting

### 5. 🧠 Feedback loop (auto-mejora)
- Analizar errores del modelo para identificar patrones
- Ajustar automáticamente:
  - Learning rate (reducir si oscila, aumentar si converge lento)
  - Dropout (aumentar si hay overfitting)
  - Número de capas (aumentar si hay underfitting)
- Registrar cada ajuste y su resultado en memoria.md

---

## Comandos que el agente puede ejecutar

```bash
# Pipeline completo automatizado
make all

# Pasos individuales
make preprocess   # Preprocesamiento de datos
make train        # Entrenamiento del modelo
make evaluate     # Evaluación y reportes
make reset        # Limpiar resultados anteriores

# Dependencias
make install      # pip install -r requirements.txt
```

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
  
SI resultado mejora consistentemente → Continuar entrenando
```

---

## Convenciones de uso

- Leer siempre `memoria.md` antes de iniciar una nueva tarea.
- Registrar cada acción ejecutada en `memoria.md`.
- Usar nombres descriptivos para modelos y outputs.
- Si no se puede decidir automáticamente, documentar la duda en memoria.md para revisión humana.
