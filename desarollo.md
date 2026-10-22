# Desarrollo

Documento de referencia para quien continúe el proyecto: qué hay en cada
carpeta, cómo se ejecuta y por qué está organizado así.

## 1. Visión general

El repositorio mezcla dos responsabilidades que se mantienen separadas a
propósito:

| Parte | Ubicación | Responsabilidad |
| --- | --- | --- |
| Entrenamiento | `colab/py/red_neuronal_ganado.py` | Preparar datos, entrenar, evaluar y generar el checkpoint |
| API | `backend/` | Cargar el checkpoint y servir predicciones |
| Interfaz | `frontend/` | Subir la imagen y mostrar las razas con su confianza |
| Artefactos | `data/NN/cattle_training/` | Checkpoint y resultados de la última ejecución |
| Imágenes | `data/NN/cattle_dataset/` | Dataset original (no versionado) |

El script de `colab/` es la fuente de verdad del modelo y **no debe
modificarse** para adaptarlo al backend. El backend tiene su propia copia del
código de inferencia. Si el entrenamiento cambia la arquitectura o el formato
del checkpoint, hay que actualizar `backend/src/cattle_api/inference.py` de
forma coordinada.

## 2. Estructura

```text
redes-neuronales/
├── config.yaml                     # Configuración del experimento (entrenamiento)
├── pyproject.toml                  # Dependencias del proyecto ML + pytest
├── colab/py/red_neuronal_ganado.py # Entrenamiento (no tocar desde el backend)
├── data/NN/
│   ├── cattle_dataset/             # Imágenes por raza, una carpeta por clase
│   └── cattle_training/
│       ├── best_cattle_breed_model.pth
│       ├── evaluation.json
│       ├── confusion_matrix.png
│       ├── misclassifications.json
│       ├── misclassified_images.png
│       └── training_history.csv
├── backend/
│   ├── pyproject.toml              # Dependencias de runtime de la API
│   └── src/cattle_api/
│       ├── config.py               # Rutas, dispositivo, CORS, límites
│       ├── schemas.py              # Modelos Pydantic de entrada/salida
│       ├── inference.py            # Carga del checkpoint y predicción
│       └── main.py                 # FastAPI, endpoints y validación de subidas
├── frontend/
│   ├── app/                        # Layout, página y estilos globales
│   ├── components/                 # Subida de imagen y resultados
│   └── lib/api.ts                  # Cliente HTTP y nombres legibles de razas
└── tests/test_inference.py         # Carga real del checkpoint y prueba de la API
```

## 3. Modelo y checkpoint

- Arquitectura: `efficientnet_b0` (torchvision), con la capa final reemplazada
  por una `Linear` del número de razas.
- Clases actuales: `ayrshire-cattle`, `brown-swiss-cattle`,
  `holstein-friesian-cattle`, `jersey-cattle`, `red-dane-cattle`.
- El checkpoint es un diccionario con `model_state_dict`, `class_names`,
  `model_name`, `image_size`, `mean`, `std`, `best_epoch`, `val_macro_f1` y
  `seed`.

`inference.py` valida ese contrato antes de cargar los pesos: comprueba que
`model_name` sea `efficientnet_b0`, que `class_names` sea una lista de strings
y que exista `model_state_dict`. Si el checkpoint no cumple, falla al arrancar
en vez de predecir mal.

Las probabilidades se calculan con `softmax` y se devuelven redondeadas a 4
decimales como lista ordenada de mayor a menor.

## 4. Backend

### Arranque

```bash
uv pip install --python .venv/bin/python -e backend
.venv/bin/uvicorn cattle_api.main:app --reload --port 8000
```

El paquete se instala en modo editable en el entorno existente: no se crea un
segundo `.venv`. Las dependencias de la API están declaradas en
`backend/pyproject.toml`, así que el backend podría desplegarse por separado
sin arrastrar las dependencias de entrenamiento.

### Endpoints

| Método | Ruta | Descripción |
| --- | --- | --- |
| `GET` | `/health` | Estado, dispositivo y clases cargadas |
| `POST` | `/api/v1/predictions` | Predicción a partir de un archivo `file` |

Ejemplo:

```bash
curl -F "file=@foto.jpg;type=image/jpeg" \
  http://localhost:8000/api/v1/predictions
```

Respuesta:

```json
{
  "model": "efficientnet_b0",
  "predictions": [
    { "breed": "jersey-cattle", "confidence": 0.9329 },
    { "breed": "red-dane-cattle", "confidence": 0.0584 }
  ]
}
```

Códigos de error: `400` archivo vacío o imagen ilegible, `413` supera el
límite de tamaño, `415` el contenido no es una imagen.

Ejemplo puntual:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/predictions \
  -F "file=@/home/jsmor/dev/inst/redes-neuronales/data/NN/cattle_dataset/jersey-cattle/Jerseycattle147.jpg;type=image/jpeg" \
  | jq .
```

### Configuración

`backend/src/cattle_api/config.py` resuelve todas las rutas desde la ubicación
del propio archivo (`parents[3]`), nunca desde el directorio de trabajo. Así
`uvicorn` puede lanzarse desde cualquier carpeta.

| Variable | Por defecto | Uso |
| --- | --- | --- |
| `CATTLE_CHECKPOINT` | `data/NN/cattle_training/best_cattle_breed_model.pth` | Ruta del `.pth` |
| `CATTLE_DEVICE` | automático (`cuda` si existe) | Forzar `cpu` o `cuda` |
| `CATTLE_TOP_K` | `3` | Número de razas devueltas |
| `CATTLE_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Orígenes permitidos, separados por coma |
| `CATTLE_MAX_UPLOAD_BYTES` | `10485760` (10 MB) | Tamaño máximo de subida |

### Ciclo de vida

El modelo se carga **una sola vez** en el evento `lifespan` de FastAPI y se
guarda en `app.state.engine`. La predicción es síncrona y se ejecuta en el
threadpool (`run_in_threadpool`) para no bloquear el bucle de eventos.

### Seguridad básica

- El backend nunca expone la ruta del checkpoint ni el estado interno del
  modelo; solo etiquetas y probabilidades.
- Se valida `content_type`, tamaño y decodificación antes de tocar la red.
- La imagen se decodifica en memoria (`BytesIO`), no se escribe en disco.
- CORS está restringido a los orígenes configurados.

## 5. Frontend

Next.js (App Router) con React 19 y TypeScript. Sin framework de CSS: los
estilos están en `app/globals.css`.

```bash
cd frontend
bun install        # o npm install
bun run dev        # http://localhost:3000
bun run build      # verificación de tipos y build de producción
```

- `NEXT_PUBLIC_API_URL` en `.env.local` apunta a la API (por defecto
  `http://localhost:8000`). `.env.example` sirve de plantilla.
- `components/image-upload.tsx`: arrastrar y soltar, selección por diálogo,
  vista previa y validación local de tipo y tamaño.
- `components/prediction-results.tsx`: raza principal, barra de confianza y
  alternativas.
- `lib/api.ts`: única puerta hacia la API; también traduce los nombres de
  carpeta (`jersey-cattle`) a etiquetas legibles (`Jersey`).

El navegador nunca recibe el modelo ni el checkpoint: solo el JSON con las
predicciones.

## 6. Tests

```bash
.venv/bin/python -m pytest tests -q
```

`tests/test_inference.py` usa el checkpoint y una imagen reales del dataset:

- carga del checkpoint y predicción coherente con la etiqueta conocida;
- `POST /api/v1/predictions` a través de `TestClient`;
- rechazo de un archivo que no es imagen.

Si faltan el checkpoint o el dataset, los tests se omiten con `skipif`.
`pyproject.toml` añade `backend/src` a `pythonpath`, así que no hace falta
instalar el paquete para ejecutarlos.

## 7. Decisiones de diseño

1. **Backend independiente del entrenamiento.** `inference.py` solo importa
   `torch`, `torchvision` y `PIL`. No importa el script de `colab/` ni prepara
   `DataLoader`s.
2. **Dependencias separadas, un solo entorno.** `backend/pyproject.toml`
   declara lo que necesita la API; se instala en el `.venv` existente para
   evitar duplicar varios GB de PyTorch. Crear un entorno aparte solo tiene
   sentido para desplegar el servicio en otra máquina.
3. **Rutas desde el código, no desde el shell.** El checkpoint se localiza
   respecto al repositorio, con opción de sobrescribirlo por variable de
   entorno.
4. **El checkpoint no se sube al navegador.** Las predicciones viajan como
   JSON; el modelo permanece en el servidor.

## 8. Flujo de trabajo típico

1. Entrenar en Colab o localmente con `colab/py/red_neuronal_ganado.py`.
2. Copiar el nuevo `best_cattle_breed_model.pth` a
   `data/NN/cattle_training/`.
3. Arrancar la API y comprobar `GET /health`: debe listar las clases nuevas.
4. Arrancar el frontend y probar una subida.
5. Ejecutar `pytest` antes de dar el cambio por terminado.

## 9. Limitaciones conocidas

- El script de entrenamiento no comparte código con el backend; un cambio de
  arquitectura exige editar ambos.
- `TOP_K` no valida contra el número real de clases, se recorta con `min()`.
- No hay autenticación: está pensado para una red local o un grupo reducido.
- `data/NN/cattle_dataset/` está ignorado por Git; sin él, los tests de imagen
  se omiten.
