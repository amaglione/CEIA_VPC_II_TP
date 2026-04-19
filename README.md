# Trabajo Práctico — Visión por Computadora II
### CEIA · Universidad de Buenos Aires

**Autor:** Agustín Maglione | Juan Chunga | Juan Cruz Becerra

**Materia:** Visión por Computadora II  
**Año:** 2026

---

## Descripción General

Este trabajo implementa y compara dos enfoques de visión por computadora para la **detección, clasificación y segmentación automática de tumores de mama en imágenes de ecografía**. El problema se enmarca en el dataset BUSI (*Breast Ultrasound Images*), con tres clases clínicas: `benign`, `malignant` y `normal`.

Se desarrollan dos pipelines independientes:

| Enfoque | Arquitectura | Tarea |
|---|---|---|
| **Pipeline YOLO** | YOLOv11-nano (Stage 1 - Definición de Existencia de Lesión) + YOLOv11m-seg (Stage 2 - Clasificación de Lesión) | Detección + segmentación de instancias en dos etapas |
| **UNet++ multi-tarea** | UNet++ con encoder ResNet34 | Segmentación binaria + clasificación 3 clases en un único modelo |

---

## Estructura del Repositorio

```
CEIA_VPC_II_TP/
│
├── 1. EDA.ipynb                        # Análisis exploratorio del dataset
├── 2. yolov11m_pipeline_v8.ipynb       # Entrenamiento del pipeline YOLO de dos etapas
├── 3. yolo_validation.ipynb            # Validación e inferencia del pipeline YOLO
├── 4. Unet++ (Paper).ipynb             # Modelo UNet++ multi-tarea con CV de 11 folds
│
├── Unet++ (Paper) - Model Backup/      # https://drive.google.com/drive/folders/1j7yFy4S7ke93j5jyUimNFHSg0K4X4eq6?usp=drive_link
│   ├── fold_results.csv                # Métricas por fold (Dice, IoU, Acc, AUROC)
│   └── unetpp_best_final.pt            # Pesos del mejor fold (~105 MB)
│
└── Yolo - Model Backup/                # https://drive.google.com/file/d/1FRQ7sWiKGqpwMVek8y0-LBwME1bLXEAW/view?usp=drive_link
    ├── models_busi_pipeline_v8/
    │   ├── stage1_yolo11_v8.pt         # Pesos del detector (Stage 1)
    │   └── stage2_yolo11_seg_v8.pt     # Pesos del segmentador (Stage 2)
    └── runs/
        ├── detect/busi_stage1_v8/      # Artefactos de entrenamiento Stage 1
        └── segment/busi_stage2_v8/     # Artefactos de entrenamiento Stage 2
```

---

## Introducción selección de dataset

Inicialmente, consideramos el conjunto de datos **Breast Ultrasound Images Dataset** disponible en Kaggle, presentado por Al-Dhabyani et al. Este dataset contiene imágenes de ultrasonido categorizadas en tres clases: benignas, malignas y normales, junto con sus correspondientes máscaras de lesiones.

Sin embargo, tras revisar un análisis posterior realizado por Pawłowska et al., se identificaron varios problemas en el conjunto de datos original, incluyendo inconsistencias en las anotaciones y posibles errores de etiquetado. Estas limitaciones podrían afectar negativamente el entrenamiento y la evaluación del modelo.

Por esta razón, decidimos migrar a una versión corregida del conjunto de datos, denominada **BUSI-Corrected**, propuesta por Tasnim y Hasan. Esta versión aborda las inconsistencias reportadas en el dataset original y proporciona una calidad de anotación mejorada, lo que la hace más adecuada para el entrenamiento de modelos de deep learning confiables.

El conjunto de datos final utilizado en este estudio consiste en imágenes médicas categorizadas en tres clases: benignas, malignas y normales. Cada imagen puede incluir una o más máscaras anotadas que indican la ubicación de las lesiones. Debido a la naturaleza de los conjuntos de datos médicos, la distribución de las muestras es inherentemente desbalanceada, con una mayor proporción de imágenes con lesiones en comparación con los casos normales.

El conjunto de datos contiene un total de 750 imágenes de ultrasonido, distribuidas de la siguiente manera: 410 muestras benignas, 179 malignas y 161 normales. Se proporcionan las máscaras de segmentación correspondientes para las imágenes con lesiones (casos benignos y malignos).

### Dataset utilizado — BUSI Corrected

Se utilizó la versión corregida disponible en Kaggle (`jarintasnim090/busi-corrected`):

| Clase | Imágenes |
|---|---|
| `benign` | 410 |
| `malignant` | 179 |
| `normal` | 161 |
| **Total** | **750** |

El dataset presenta **desbalance de clases** (`benign:malignant ≈ 2.3:1`), abordado con estrategias de augmentación diferenciadas.

> **Referencia:** Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. *Dataset of breast ultrasound images*. Data in Brief. 2020 Feb;28:104863.

---

## Notebook 1 — Análisis Exploratorio (EDA)

El EDA cubre:

- Verificación de formatos y modos de color (todas las imágenes son PNG, mayoritariamente en escala de grises).
- Distribución de tamaños de imagen por clase (histogramas y scatter de width × height).
- Análisis de espacio de color RGB/HSV (scatter 3D, dominantes de Hue).
- **Detección de duplicados** mediante perceptual hash (`imagehash.phash`):
  - Duplicados exactos (hamming = 0): 82 pares detectados.
  - Duplicados cruzados entre clases (benign ↔ malignant, etc.).
- Visualización de muestras con sus máscaras por clase.
- **Decisión final:** usar BUSI Corrected y abordar el problema como **segmentación de instancias** (no segmentación semántica) para distinguir lesiones individuales.
- Plan preliminar de augmentación para imágenes médicas: rotaciones suaves (±5°), traslación/zoom, CLAHE, ruido speckle/Gaussiano, deformación elástica.

---

## Notebook 2 — Pipeline YOLOv11 de Dos Etapas

### Arquitectura

```
Imagen de entrada
       │
       ▼
┌──────────────────┐
│  Stage 1: Detector│  yolo11n  (detección binaria: lesion / normal)
│  Conf threshold   │  640×640 px
└──────────────────┘
       │  sin lesión → clase "normal"
       │  con lesión ↓
┌──────────────────────┐
│  Stage 2: Segmentador│  yolo11m-seg  (segmentación: benign / malignant)
│  Conf threshold      │  768×768 px
└──────────────────────┘
       │
       ▼
  Máscara + clase final
```

### Preparación del Dataset

**Stage 1 — Detección (lesión vs normal):**
- Split 70/15/15 estratificado (seed=42).
- Train forzado a 50/50 mediante augmentación offline de imágenes `normal`.
- Val/test mantienen proporciones naturales (corrección respecto a versión v7 que submuestraba incorrectamente).
- Bounding box de lesiones: unión de todas las máscaras. BBox de normales: imagen completa.

**Stage 2 — Segmentación (benign vs malignant):**
- Solo imágenes con lesión (`normal` excluida).
- Train balanceado a `{benign: 190, malignant: 300}`.
- **Copy-paste augmentation con Poisson blending** (`seamlessClone`): pega lesiones malignas sobre fondos `normal` para generar muestras sintéticas de malignant (prob=0.28).
- Polígonos YOLO extraídos con `cv2.findContours` + `approxPolyDP` (eps=0.0005·arcLength).

### Entrenamiento

| | Stage 1 | Stage 2 |
|---|---|---|
| Modelo base | yolo11n.pt | yolo11m-seg.pt |
| Epochs | 200 | 250 |
| Imgsz | 640 | 768 |
| Batch | 64 | 32 |
| Optimizer | AdamW | AdamW |
| lr0 | 0.002 | 0.002 |
| LR schedule | CosineAnnealingLR | CosineAnnealingLR |
| Dropout | — | 0.1 |
| `cls` weight | 1.5 | 2.0 |
| `multi_scale` | False* | — |

> \* `multi_scale=True` producía `ZeroDivisionError` con batch=64; se deshabilitó explícitamente.

### Evaluación del Pipeline

- Búsqueda de umbrales óptimos (`conf_s1 × conf_s2`) minimizando la función:
  ```
  score = 0.50 · sensitivity_malignant + 0.30 · specificity_normal + 0.20 · sensitivity_benign
  ```
- Umbrales óptimos v8: `conf_s1 = 0.25`, `conf_s2 = 0.15`.
- Métricas por etapa: mAP50, mAP50-95, precision, recall (box y mask).

**Resultados Stage 1 (mejor época ~150):**
- Box mAP50 ≈ **0.94** | Box mAP50-95 ≈ **0.78**

**Resultados Stage 2 (mejor época ~206):**
- Box mAP50 ≈ **0.81** | Mask mAP50 ≈ **0.81** | Mask mAP50-95 ≈ **0.53**

---

## Notebook 3 — Validación YOLO

Notebook autónomo para reproducir la evaluación desde una sesión limpia de Colab:

- Reconstruye los datasets con seed=42 (splits idénticos).
- Carga los pesos `stage1_yolo11_v8.pt` y `stage2_yolo11_seg_v8.pt`.
- Ejecuta `.val()` de Ultralytics en val y test para ambas etapas.
- Mide **latencia de inferencia** (Stage 1 solo, delta Stage 2, pipeline completo, FPS) sobre 20 imágenes de validación.
- Genera figura resumen con tres paneles: mAP50 por clase, F1-score por clase, latencia por etapa.

---

## Notebook 4 — UNet++ Multi-Tarea

### Arquitectura

Basado en Hesaraki et al. (2025) — *"Breast cancer ultrasound image segmentation using improved 3D UNet++"*, adaptado a 2D.

```
Imagen (256×256×1)
       │
       ▼
┌─────────────────────────────────────────┐
│  Encoder: ResNet34 (pesos ImageNet)     │
│  Cargados manualmente desde torchvision │
└─────────────────────────────────────────┘
       │  feature maps en múltiples escalas
       ▼
┌─────────────────────────────────────────┐
│  Decoder: UNet++ con dense skip paths   │
│  (segmentation_models_pytorch)          │
└─────────────────────────────────────────┘
       │                    │
       ▼                    ▼
┌──────────────┐  ┌──────────────────────────┐
│ Seg. Head    │  │ Classification Head       │
│ (1 canal,    │  │ GAP → FC(512→256) → ReLU │
│  logits,     │  │ → FC(256→3) + Dropout 0.5│
│  binario)    │  │ (benign/malignant/normal) │
└──────────────┘  └──────────────────────────┘
```

### Loss Function y Optimización

```
L = 0.5 · (DiceLoss + BCEWithLogitsLoss) + 0.5 · CrossEntropyLoss
```

| Hiperparámetro | Valor |
|---|---|
| Optimizer | Adam |
| Learning rate | 5e-5 |
| Weight decay | 1e-5 |
| LR schedule | CosineAnnealingLR (T_max=100, eta_min=5e-6) |
| Batch size | 32 |
| Image size | 256×256 |
| Epochs por fold | 100 |
| Early stopping | patience=15 (val Dice) |

### Protocolo de Entrenamiento

- **11-fold Stratified Cross-Validation** (alineado con el paper de referencia).
- `WeightedRandomSampler` por fold para balancear clases en cada batch (sin oversampling offline).
- Augmentación: CLAHE (aplicado universalmente, en train y val), HorizontalFlip, Affine, ElasticTransform, RandomBrightnessContrast, GaussNoise, GaussianBlur.
- Normalización ImageNet; `ToTensorV2` con sincronización imagen/máscara.
- Checkpoint del mejor fold guardado como `unetpp_best_final.pt`.

### Resultados (11 folds)

| Métrica | Mín | Máx | **Media** | Std |
|---|---|---|---|---|
| Dice | 0.7802 | 0.8864 | **0.8454** | 0.035 |
| IoU | — | — | **0.7745** | — |
| Accuracy | — | — | **0.934** | — |
| AUROC | — | — | **0.986** | — |

> **Referencia del paper (Hesaraki et al., 2025):** Dice UNet++ ∈ [0.4849, 0.6716] — los resultados de este trabajo superan el baseline reportado.

---

## Comparación entre Enfoques

El objetivo central del trabajo es contrastar el pipeline YOLO de dos etapas con el modelo UNet++ multi-tarea. La siguiente tabla resume las diferencias arquitectónicas y de metodología más relevantes:

| Aspecto | Pipeline YOLO | UNet++ Multi-Tarea |
|---|---|---|
| **Paradigma** | Two-stage: detección → segmentación | Single-stage: segmentación + clasificación simultánea |
| **Número de modelos** | 2 (detector + segmentador) | 1 (encoder-decoder multi-task) |
| **Backbone** | COCO-pretrained (YOLOv11) | ImageNet-pretrained ResNet34 |
| **Resolución de entrada** | 640 / 768 px | 256×256 px |
| **Tipo de segmentación** | Instancias (polígonos) | Binaria pixel-wise (máscara) |
| **Clasificación** | Implícita (clase del bbox) | Head dedicada (3 clases) |
| **Balanceo de clases** | Copy-paste + augmentación offline | WeightedRandomSampler por fold |
| **Evaluación** | Sweep de umbrales + métricas pipeline | 11-fold Stratified Cross-Validation |
| **Métricas principales** | mAP50, Sensitivity, Specificity | Dice, IoU, Accuracy, AUROC |
| **Stage 1 mAP50** | **0.94** (lesión vs normal) | N/A |
| **Dice / Mask mAP50** | 0.81 (stage 2) | **0.8454** (promedio 11 folds) |
| **AUROC clasificación** | — | **0.986** |
| **Interpretabilidad** | Alta (BBox + máscara por instancia) | Media (máscara binaria + logits) |
| **Inferencia** | Secuencial (latencia acumulada) | Una sola pasada forward |
| **Complejidad de entrenamiento** | Alta (2 datasets, 2 modelos, sweep) | Media (CV automatizado, 1 modelo) |

### Ventajas y Limitaciones

**Pipeline YOLO:**
- Permite rechazar explícitamente imágenes sin lesión (Stage 1), reduciendo falsos positivos en la clasificación.
- La separación en dos etapas permite optimizar cada subproblema independientemente.
- El sweep de umbrales brinda control clínico sobre el trade-off sensibilidad/especificidad.
- Mayor complejidad de implementación y más propenso a errores de integración entre etapas.

**UNet++ Multi-Tarea:**
- Arquitectura más limpia y reproducible gracias a la cross-validation estandarizada.
- Mejor Dice en segmentación y AUROC de clasificación.
- Aprende representaciones compartidas entre segmentación y clasificación.
- Menos control sobre el umbral de decisión clínica entre clases.

---

## Dependencias

Las dependencias se instalan inline en cada notebook. No hay `requirements.txt` —  los notebooks asumen entorno Google Colab.

| Notebook | Dependencias principales |
|---|---|
| 1 (EDA) | `kagglehub`, `imagehash`, `opencv-python`, `matplotlib`, `seaborn` |
| 2 & 3 (YOLO) | `ultralytics`, `opencv-python`, `albumentations`, `pyyaml` |
| 4 (UNet++) | `segmentation-models-pytorch`, `albumentations`, `scikit-learn`, `torch`, `torchvision` |

> **Nota:** `albumentations` cambió su API entre v1.x y v2.x (e.g., `GaussNoise` pasó de `var_limit` a `std_range/mean_range`). El código usa la API de v2. Ejecutar en un entorno con v1 puede producir errores.

---

## Reproducción

### Pipeline YOLO

1. Abrir `2. yolov11m_pipeline_v8.ipynb` en Google Colab.
2. Montar Google Drive y configurar la ruta de backup.
3. Ejecutar todas las celdas en orden (el notebook descarga el dataset de Kaggle automáticamente).
4. Los pesos se guardan en `/content/models_busi_pipeline_v8/`.
5. Para validación independiente, usar `3. yolo_validation.ipynb` apuntando al zip de backup.

### UNet++ Multi-Tarea

1. Abrir `4. Unet++ (Paper).ipynb` en Google Colab (se recomienda GPU A100 o similar).
2. El notebook incluye un **Web-Worker keep-alive** en JavaScript para evitar desconexiones durante las ~11 horas de entrenamiento.
3. Cada fold descarga automáticamente su checkpoint vía `colab_files.download`.
4. El mejor checkpoint global se guarda como `unetpp_best_final.pt`.

---

## Referencias

- Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. *Dataset of breast ultrasound images*. Data in Brief. 2020 Feb;28:104863. doi:10.1016/j.dib.2019.104863
- Pawłowska A, et al. *Curated benchmark dataset for ultrasound based breast lesion analysis*. Scientific Data. 2023.
- Hesaraki S, et al. *Breast cancer ultrasound image segmentation using improved 3D UNet++*. 2025.
- Jocher G, et al. *Ultralytics YOLO11*. 2024. [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- Iakubovskii P. *Segmentation Models PyTorch*. [https://github.com/qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- Zhou Z, et al. *UNet++: A Nested U-Net Architecture for Medical Image Segmentation*. DLMIA 2018.

---

## Pendientes y Mejoras para Revisar con el Grupo

> Lista generada automáticamente para discutir en la próxima reunión.

### Bugs confirmados (afectan reproducibilidad)

| # | Dónde | Problema | Estado |
|---|---|---|---|
| 1 | NB2 celda 29 | Los nombres de los modelos exportados (`yolo11s`, `yolo11l-seg`) no coinciden con los entrenados (`yolo11n`, `yolo11m-seg`). El NB3 no puede cargarlos sin renombramiento manual. | ⚠️ Pendiente |
| 2 | NB4 `cell-drive-mount` y `cell-17` | `weight_only=False` (typo) en vez de `weights_only=False` — el argumento era ignorado y el checkpoint fallaba con `UnpicklingError`. | ✅ Corregido |

### Issues metodológicos (para discutir)

| # | Dónde | Problema | Impacto |
|---|---|---|---|
| 3 | NB2 `results.csv` Stage 1 | NaNs en épocas 6, 7, 9 y 10 de `val/box_loss`, `val/cls_loss`, `val/dfl_loss`. Probable inestabilidad de LR en warm-up (`lr0=0.002` + `batch=64`). Convergió igual, pero es una señal de warning. | Bajo |
| 4 | NB4 `build_transforms` | CLAHE con `clip_limit=(1.5, 3.0)` aleatorio en train vs `clip_limit=2.0` fijo en val. Leve discrepancia distribucional entre split de train y de validación. | Bajo–Medio |
| 5 | NB4 celda 23 | Si el entrenamiento se corta antes del primer fold, `best_global_state = None` y `best_model.load_state_dict(None)` tira `TypeError`. No hay guard. | Bajo (edge case) |

### Deuda técnica (no crítica)

| # | Dónde | Descripción |
|---|---|---|
| 6 | NB2 + NB3 | ~800 líneas de utilidades copiadas verbatim entre ambos notebooks. Cualquier fix hay que aplicarlo en dos lugares. |
| 7 | Todos los NB | Rutas hardcodeadas a `/content/`. Ejecutar fuera de Colab requiere find-and-replace manual. |
| 8 | Todos los NB | Sin versiones pineadas de dependencias. `albumentations` v1 vs v2 rompe la API de `GaussNoise`. |
| 9 | Git | `.DS_Store` commiteados. Ya agregado al `.gitignore`, pero los existentes siguen trackeados — hacer `git rm --cached .DS_Store`. |
| 10 | Git | Los pesos `.pt` (~250 MB en total) están en el repo. Considerar Git LFS o moverlos solo a Drive. |

### Posibles mejoras para el TP (si hay tiempo)

- [ ] **Arreglar nombres de exportación** en NB2 para que coincidan con los modelos realmente entrenados (`yolo11n` y `yolo11m-seg`), o bien actualizar el entrenamiento para usar `yolo11s` y `yolo11l-seg` si era esa la intención.
- [ ] **Guard para `best_global_state`** antes de `load_state_dict` en NB4.
- [ ] **Pinear versiones** de `albumentations`, `segmentation-models-pytorch`, `ultralytics` en los `pip install` de cada notebook.
- [ ] **Refactorizar utilities de YOLO** a un archivo `utils.py` importado por NB2 y NB3 para eliminar la duplicación.
