# ============================================
# Makefile — Comandos automatizados para el agente
# ============================================

.PHONY: help install preprocess train evaluate test clean all

help:                           # Muestra ayuda con todos los comandos disponibles
	@echo "🧠 Redes Neuronales — Comandos del Agente"
	@echo ""
	@echo "  make install      Instalar dependencias"
	@echo "  make preprocess   Preprocesar dataset (data/raw/ → data/preprocessed/)"
	@echo "  make train        Entrenar modelo completo"
	@echo "  make evaluate     Evaluar último modelo entrenado"
	@echo "  make test         Ejecutar tests unitarios"
	@echo "  make clean        Limpiar resultados y modelos"
	@echo "  make all          Pipeline completo: preprocess → train → evaluate"

install:                        # pip install -r requirements.txt
	pip install -r requirements.txt

preprocess:                     # Ejecuta el preprocesamiento de datos
	python src/dataset.py --mode preprocess

train:                          # Entrena el modelo según config.yaml
	python src/train.py

evaluate:                       # Evalúa el mejor modelo disponible
	python src/evaluate.py --model models/best_model.keras

test:                           # Ejecuta tests con pytest
	pytest tests/ -v --cov=src

clean:                          # Elimina modelos y outputs generados
	rm -rf output/*
	rm -f models/*.keras models/*.h5 models/*.pt
	@echo "✅ Limpieza completada"

all: preprocess train evaluate   # Pipeline completo automatizado
	@echo ""
	@echo "🎉 Pipeline completo ejecutado exitosamente"
	@echo "📊 Revisa los resultados en output/"
