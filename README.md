## Project Structure

```
--- Classification_FM_with_GNN.py     # GNN-based classification pipeline
--- Retrieval_with_GNN.py            # Image retrieval using segmented embeddings
--- generate_gradcams.py             # Grad-CAM visualization for interpretability
--- service.py                       # FastAPI inference & retrieval service
--- config.py                        # Configuration (paths, hyperparameters)
--- vision/                          # Optional: modular package (future)
--- logs/                            # Training & retrieval logs
--- gradcam/                         # Generated heatmaps
--- static/                          # Served heatmaps (API)
--- datasets/                        # (User-provided) BACH dataset
--- README.md
```

> **Note**: Additional files (e.g., `utils.py`, `models/`, `data/`) can be added later without breaking the pipeline.

---

## Key Features

| Feature | File | Description |
|-------|------|-----------|
| **GNN Classification** | `Classification_FM_with_GNN.py` | Builds graphs from k-means segmented regions, extracts ViT patch embeddings, trains GAT model |
| **Image Retrieval** | `Retrieval_with_GNN.py` | Uses mean-pooled segment features for FAISS indexing and fast similarity search |
| **Interpretability** | `generate_gradcams.py` | Generates class-specific Grad-CAM heatmaps overlaid on original tissue images |
| **Inference API** | `service.py` | FastAPI service for real-time classification + retrieval + visualization |
| **Reproducibility** | All | Fixed seeds, stratified splits, k-fold CV, detailed logging |

---

## API Service (`service.py`)

A **production-ready FastAPI backend** that serves:

- **Classification** with GNN
- **Grad-CAM heatmap** generation
- **Similar case retrieval** via FAISS

### API Endpoints

#### 1. `POST /analyze` – Classify & Visualize

**Upload a `.tif` image → get prediction + heatmap**

> Heatmap available at: `http://localhost:8000/static/heatmap_...png`

---

#### 2. `GET /neighbors/{case_id}` – Retrieve Similar Cases

**Find top-k similar cases from gallery**

```bash
curl "http://localhost:8000/neighbors/42?k=5"
```
---

### How the API Works (Internals)

| Step | Component |
|------|---------|
| 1. Startup | Loads ViT, GNN, FAISS index, gallery metadata |
| 2. `/analyze` | → Segment → Build graph → GNN → Grad-CAM → Save heatmap |
| 3. `/neighbors` | → Query FAISS → Return top-k matches |

> **Zero-downtime**: Index built once at startup.  
> **Scalable**: Add GPU, async, or Redis caching later.

---

## Code Overview

### `service.py`

| Endpoint | Function |
|--------|---------|
| `@app.on_event("startup")` | Initializes models, FAISS index, gallery |
| `POST /analyze` | Full pipeline: graph → GNN → heatmap |
| `GET /neighbors/{case_id}` | FAISS nearest neighbor search |

### Docker

```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "service.py"]
```

## Next Steps
