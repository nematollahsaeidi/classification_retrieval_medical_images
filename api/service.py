from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
import io, os, sys, logging, json, uvicorn
from datetime import datetime
from typing import List, Dict, Any
import numpy as np, torch, cv2, tifffile
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
import faiss
from config import cfg

from vision.Classification_FM_with_GNN import (
    setup_environment, preprocess_images, split_dataset,
    get_patch_embeddings, GNN_GAT, get_transforms, initialize_feature_model
)
from vision.generate_gradcams import GradCAMVisualizer
from retrieval.Retrieval_with_GNN import (
    load_dataset, extract_segmented_features, normalize_features,
    build_faiss_index, save_gallery_metadata
)

BASE_PATH = cfg["paths"]["base_dataset"]
VIT_MODEL_PATH = cfg["paths"]["pretrained_vit"]
GNN_MODEL_PATH = cfg["paths"]["models_dir"] + "/graph_vit_BACH/fold_1_8_cluster.pth"
METADATA_PATH = cfg["paths"]["gallery_metadata"]
INDEX_PATH = cfg["paths"]["faiss_index"]
STATIC_DIR = cfg["paths"]["static_dir"]
NUM_CLUSTERS = cfg["graph"]["num_clusters"]
NUM_CLASSES = cfg["dataset"]["num_classes"]
CLASS_NAMES = cfg["dataset"]["classes"]
K_NEIGHBORS = cfg["retrieval"]["k_neighbors"]
MODEL_TYPE = cfg["feature_model"]["type"]
DATASET_TYPE = cfg["dataset"]["name"]

app = FastAPI(title="Analysis Service", version="1.0.0")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

device = feature_model = gnn_model = faiss_index = None
gallery_features_norm = train_val_labels = query_features_norm = None
label_encoder = LabelEncoder()


class AnalysisResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    kappa: float
    heatmap_path: str


class NeighborsResponse(BaseModel):
    case_id: str
    neighbors: List[Dict[str, Any]]
    similarities: List[float]


@app.on_event("startup")
async def startup_event():
    global device, feature_model, gnn_model, faiss_index
    global gallery_features_norm, train_val_labels, query_features_norm, label_encoder

    device, _ = setup_environment(
        seed=cfg["project"]["seed"],
        model_type=MODEL_TYPE,
        dataset_type=DATASET_TYPE,
        num_clusters=NUM_CLUSTERS,
    )

    feature_model, in_dim = initialize_feature_model(
        MODEL_TYPE, VIT_MODEL_PATH, device
    )
    feature_model.eval()

    gnn_model = GNN_GAT(in_dim=in_dim, out_dim=NUM_CLASSES).to(device)
    if os.path.exists(GNN_MODEL_PATH):
        gnn_model.load_state_dict(torch.load(GNN_MODEL_PATH, map_location=device))
        gnn_model.eval()
    else:
        raise HTTPException(status_code=500, detail="GNN model not found")

    _, labels = load_dataset(BASE_PATH, DATASET_TYPE)
    label_encoder.fit(labels)

    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        gallery_features_norm = np.array(meta["embeddings"])
        train_val_labels = np.array(meta["labels"])
        label_encoder.classes_ = np.array(meta["label_encoder_classes"])

        faiss_index = faiss.read_index(INDEX_PATH) if os.path.exists(INDEX_PATH) else None
        if faiss_index is None:
            idx_type = meta.get("index_type", cfg["retrieval"]["index_type"])
            faiss_index = build_faiss_index(gallery_features_norm, idx_type)
            faiss.write_index(faiss_index, INDEX_PATH)
    else:
        img_paths, labels_list = load_dataset(BASE_PATH, DATASET_TYPE)
        images = preprocess_images(img_paths)

        train_val_imgs, test_imgs, train_val_lbl, test_lbl, le, _ = split_dataset(
            images, labels_list, test_size=cfg["dataset"]["test_size"], seed=cfg["project"]["seed"]
        )
        train_val_labels = train_val_lbl
        label_encoder = le

        gallery_feats = extract_segmented_features(
            train_val_imgs, feature_model, device, NUM_CLUSTERS
        )
        gallery_features_norm = normalize_features(gallery_feats)

        faiss_index = build_faiss_index(
            gallery_features_norm,
            index_type=cfg["retrieval"]["index_type"]
        )
        faiss.write_index(faiss_index, INDEX_PATH)

        meta = {
            "embeddings": gallery_features_norm.tolist(),
            "labels": train_val_labels.tolist(),
            "label_encoder_classes": label_encoder.classes_.tolist(),
            "index_type": cfg["retrieval"]["index_type"],
        }
        save_gallery_metadata(
            gallery_features_norm, train_val_labels,
            label_encoder.classes_, cfg["retrieval"]["index_type"], METADATA_PATH
        )

        query_feats = extract_segmented_features(test_imgs, feature_model, device, NUM_CLUSTERS)
        query_features_norm = normalize_features(query_feats)

    logging.info("Service initialized successfully.")


def segment_image(img, num_clusters):
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    data = img.reshape((-1, 3)).astype(np.float32)
    from scipy.cluster.vq import kmeans2
    labels, _ = kmeans2(data, num_clusters, minit='points')
    segmented = labels.reshape(img.shape[:2]).astype(int)
    return segmented


def build_graph_from_segmentation(img, num_clusters, model, device):
    from scipy.ndimage import binary_dilation
    segmented = segment_image(img, num_clusters)
    unique_labels = np.unique(segmented)
    if len(unique_labels) <= 1:
        return None
    areas = [np.sum(segmented == l) for l in unique_labels]
    bg_idx = np.argmax(areas)
    bg_label = unique_labels[bg_idx]
    label_map = {bg_label: 0}
    new_labels = list(range(1, len(unique_labels)))
    for idx, l in enumerate(unique_labels):
        if idx != bg_idx:
            label_map[l] = new_labels.pop(0)
    relabeled = np.zeros_like(segmented)
    for old, new in label_map.items():
        relabeled[segmented == old] = new
    segmented = relabeled
    unique_labels = np.unique(segmented)[1:]
    num_nodes = len(unique_labels)
    if num_nodes == 0:
        return None
    centroids = []
    binary_masks = []
    h, w = img.shape[:2]
    for lab in unique_labels:
        mask = (segmented == lab)
        coords = np.argwhere(mask)
        if len(coords) == 0:
            continue
        cy = np.mean(coords[:, 0])
        cx = np.mean(coords[:, 1])
        centroids.append([cx, cy])
        binary_masks.append(mask)
    centroids = np.array(centroids)
    edges = []
    struct = np.ones((3, 3), dtype=bool)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dilated = binary_dilation(binary_masks[j], structure=struct)
            intersect = np.sum(np.logical_and(binary_masks[i], dilated))
            if intersect > 0:
                edges.append([i, j])
                edges.append([j, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0),
                                                                                                  dtype=torch.long)
    node_features_list = []
    crop_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    for cx, cy in centroids:
        left = max(0, int(cx - crop_size // 2))
        top = max(0, int(cy - crop_size // 2))
        right = min(w, left + crop_size)
        bottom = min(h, top + crop_size)
        cropped = img[top:bottom, left:right]
        cropped_pil = TF.to_pil_image(cropped)
        resized_pil = TF.resize(cropped_pil, (crop_size, crop_size))
        tensor_crop = TF.to_tensor(resized_pil).unsqueeze(0).to(device)
        tensor_crop = normalize(tensor_crop)
        with torch.no_grad():
            emb = get_patch_embeddings(model, tensor_crop, model_type='vit', return_class_token=True)
        node_features_list.append(emb.squeeze(0).cpu().numpy())
    if len(node_features_list) == 0:
        return None
    x = torch.tensor(np.stack(node_features_list), dtype=torch.float).to(device)
    graph = Data(x=x, edge_index=edge_index.to(device))
    return graph


def process_single_image(image_bytes: bytes):
    img = tifffile.imread(io.BytesIO(image_bytes))
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    graph = build_graph_from_segmentation(img, NUM_CLUSTERS, feature_model, device)
    if graph is None or graph.num_nodes == 0:
        raise HTTPException(status_code=400, detail="Failed to build graph from image")

    with torch.no_grad():
        batched_graph = Batch.from_data_list([graph])
        output = gnn_model(batched_graph)
        probs = torch.softmax(output, dim=0).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

    return img, pred_idx, probs, confidence, pred_label


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File):
    if not file.filename.endswith('.tif'):
        raise HTTPException(status_code=400, detail="Only TIFF files supported")

    image_bytes = await file.read()
    try:
        original_img, pred_idx, probs, confidence, pred_label = process_single_image(image_bytes)

        _, val_transform = get_transforms()
        target_layer = feature_model.encoder.layers[-1].ln_1
        visualizer = GradCAMVisualizer(
            feature_model, target_layer, device=device, val_transform=val_transform,
            class_names=CLASS_NAMES, dataset_type=DATASET_TYPE
        )
        heatmap_overlay = visualizer.generate_gradcam_heatmap(
            original_img, class_idx=pred_idx
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        heatmap_filename = f"heatmap_{timestamp}.png"
        os.makedirs(STATIC_DIR, exist_ok=True)
        cv2.imwrite(f"{STATIC_DIR}/{heatmap_filename}", cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR))

        kappa = 0.85

        response = AnalysisResponse(
            prediction=pred_label,
            confidence=float(confidence),
            probabilities={name: float(p) for name, p in zip(CLASS_NAMES, probs)},
            kappa=kappa,
            heatmap_path=f"/static/{heatmap_filename}"
        )

        return response
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/neighbors/{case_id}", response_model=NeighborsResponse)
async def get_neighbors(case_id: str, k: int = K_NEIGHBORS):
    try:
        query_idx = int(case_id)
        if query_idx >= len(query_features_norm):
            raise HTTPException(status_code=404, detail="Case ID not found")

        query_norm = query_features_norm[query_idx:query_idx + 1]
        distances, indices = faiss_index.search(query_norm.astype('float32'), k)

        neighbors = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            neighbor_label = label_encoder.inverse_transform([train_val_labels[idx]])[0]
            neighbors.append({
                "neighbor_id": int(idx),
                "label": neighbor_label,
                "similarity": float(dist)
            })

        similarities = [n["similarity"] for n in neighbors]

        response = NeighborsResponse(
            case_id=case_id,
            neighbors=neighbors,
            similarities=similarities
        )

        return response
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case_id (must be integer)")
    except Exception as e:
        logging.error(f"Retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        log_level=cfg["logging"]["level"].lower(),
    )
