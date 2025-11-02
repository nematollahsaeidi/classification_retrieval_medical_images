import os, numpy as np, torch, logging, math, faiss, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as transforms
from torchvision import models
import tifffile
from tqdm import tqdm
from torchvision.transforms import functional as TF
from scipy.cluster.vq import kmeans2
from config import cfg

BASE_PATH = cfg["paths"]["base_dataset"]
MODEL_PATH = cfg["paths"]["pretrained_vit"]
NUM_CLUSTERS = cfg["graph"]["num_clusters"]
INDEX_TYPE = cfg["retrieval"]["index_type"]
K = cfg["retrieval"]["k_neighbors"]
NLIST = cfg["retrieval"]["nlist"]
M = cfg["retrieval"]["M"]
METADATA_PATH = cfg["paths"]["gallery_metadata"]
SEED = cfg["project"]["seed"]


def setup_environment(seed=42, log_file=None, model_type='vit', dataset_type='BACH'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if log_file is None:
        log_file = f'{cfg["paths"]["logs_dir"]}/{model_type}_{dataset_type}/out.log'
    logging.basicConfig(
        level=logging.INFO,
        format=cfg["logging"]["format"],
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return device, logger


def load_dataset(base_path, dataset_type='BACH'):
    if dataset_type != 'BACH':
        raise ValueError("Invalid dataset_type. Only 'BACH' is supported.")
    categories = ['benign', 'insitu', 'invasive', 'normal']
    image_paths = []
    labels = []
    for category in categories:
        folder_path = os.path.join(base_path, category)
        for img_file in os.listdir(folder_path):
            if img_file.endswith('.tif'):
                image_paths.append(os.path.join(folder_path, img_file))
                labels.append(category)
    return image_paths, labels


def preprocess_images(image_paths):
    all_images = []
    for img_path in tqdm(image_paths, desc="Preprocessing images"):
        img = tifffile.imread(img_path)
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        all_images.append(img)
    return np.array(all_images)


def split_dataset(images, labels, test_size=0.2, seed=42):
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    num_classes = len(np.unique(labels_encoded))
    train_val_images, test_images, train_val_labels, test_labels = train_test_split(
        images, labels_encoded, test_size=test_size, random_state=seed, stratify=labels_encoded
    )
    del images, labels_encoded
    return train_val_images, test_images, train_val_labels, test_labels, label_encoder, num_classes


def get_patch_embeddings(model, x, model_type='vit', return_class_token=True):
    x = model._process_input(x)
    n = x.shape[0]
    batch_class_token = model.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    x = model.encoder(x)
    if return_class_token:
        return x[:, 0]
    else:
        return x[:, 1:]


def segment_image(img, num_clusters):
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    data = img.reshape((-1, 3)).astype(np.float32)
    labels, _ = kmeans2(data, num_clusters, minit='points')
    segmented = labels.reshape(img.shape[:2]).astype(int)
    return segmented


def extract_segmented_features(images, model, device, num_clusters=8):
    model.eval()
    features_list = []
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    crop_size = 224
    for img in tqdm(images, desc="Extracting segmented features"):
        segmented = segment_image(img, num_clusters)
        unique_labels = np.unique(segmented)
        if len(unique_labels) <= 1:
            features_list.append(np.zeros(768))
            continue
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
            features_list.append(np.zeros(768))
            continue
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
        if len(centroids) == 0:
            features_list.append(np.zeros(768))
            continue
        node_features_list = []
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
            features_list.append(np.zeros(768))
            continue
        pooled_feature = np.mean(node_features_list, axis=0)
        features_list.append(pooled_feature)
    return np.array(features_list)


def normalize_features(features):
    return features / np.linalg.norm(features, axis=1, keepdims=True)


def build_faiss_index(features, index_type='Flat', nlist=100, M=32):
    d = features.shape[1]
    index = None
    if index_type == 'Flat':
        index = faiss.IndexFlatIP(d)
    elif index_type == 'IVF':
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlatIP(quantizer, d, nlist)
        index.train(features.astype('float32'))
    elif index_type == 'HNSW':
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    else:
        raise ValueError(f"Invalid index_type: {index_type}. Options: 'Flat', 'IVF', 'HNSW'")
    index.add(features.astype('float32'))
    return index


def save_gallery_metadata(features_norm, labels, label_encoder_classes, index_type, metadata_path):
    gallery_metadata = {
        'embeddings': features_norm.tolist(),
        'labels': labels.tolist(),
        'label_encoder_classes': label_encoder_classes,
        'index_type': index_type
    }
    with open(metadata_path, 'w') as f:
        json.dump(gallery_metadata, f)


def search_index(index, queries_norm, k=10):
    distances, indices = index.search(queries_norm.astype('float32'), k)
    return distances, indices


def compute_metrics(gallery_indices, test_labels, train_val_labels, k_ndcg=5, k_recall=5, k_ap=10):
    def ap_k(actual, predicted, k):
        if len(predicted) > k:
            predicted = predicted[:k]
        score = 0.0
        num_hits = 0.0
        for i, p in enumerate(predicted):
            if p in actual[:k]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        return 0.0 if num_hits == 0.0 else score / num_hits

    def recall_k(actual, predicted, k):
        if len(predicted) > k:
            predicted = predicted[:k]
        hits = any(predicted[i] in actual for i in range(k))
        return 1 if hits else 0

    def ndcg_k(actual, predicted, k):
        if len(predicted) > k:
            predicted = predicted[:k]
        dcg = 0.0
        for i, p in enumerate(predicted):
            if p in actual:
                dcg += 1.0 / math.log2(i + 2)
        idcg = 0.0
        num_rel = min(k, len(actual))
        for i in range(num_rel):
            idcg += 1.0 / math.log2(i + 2)
        return dcg / idcg if idcg > 0 else 0.0

    ndcg_scores = []
    recall_scores = []
    ap_scores = []
    num_queries = len(test_labels)
    for q_idx in range(num_queries):
        query_label = test_labels[q_idx]
        relevant_indices = [g_idx for g_idx, g_label in enumerate(train_val_labels) if g_label == query_label]
        if len(relevant_indices) == 0:
            continue
        predicted_indices = gallery_indices[q_idx]
        ndcg = ndcg_k(relevant_indices, predicted_indices, k_ndcg)
        recall = recall_k(relevant_indices, predicted_indices, k_recall)
        ap = ap_k(relevant_indices, predicted_indices, k_ap)
        ndcg_scores.append(ndcg)
        recall_scores.append(recall)
        ap_scores.append(ap)
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    avg_recall = np.mean(recall_scores) if recall_scores else 0.0
    avg_map = np.mean(ap_scores) if ap_scores else 0.0
    return avg_ndcg, avg_recall, avg_map


def run_full_pipeline(base_path, model_path=None, batch_size=32, seed=42, index_type='Flat', k=10, nlist=100, M=32,
                      metadata_path=None, log_file=None, num_clusters=8):
    device, logger = setup_environment(seed, log_file)
    image_paths, labels = load_dataset(base_path)
    images = preprocess_images(image_paths)
    train_val_images, test_images, train_val_labels, test_labels, label_encoder, num_classes = split_dataset(images,
                                                                                                             labels,
                                                                                                             seed=seed)
    model = initialize_model(model_path=model_path)
    model = model.to(device)
    features_gallery = extract_segmented_features(train_val_images, model, device, num_clusters)
    features_queries = extract_segmented_features(test_images, model, device, num_clusters)
    features_gallery_norm = normalize_features(features_gallery)
    features_queries_norm = normalize_features(features_queries)
    index = build_faiss_index(features_gallery_norm, index_type, nlist, M)
    if metadata_path is None:
        metadata_path = METADATA_PATH
    save_gallery_metadata(features_gallery_norm, train_val_labels, label_encoder.classes_.tolist(), index_type,
                          metadata_path)
    distances, gallery_indices = search_index(index, features_queries_norm, k)
    avg_ndcg, avg_recall, avg_map = compute_metrics(gallery_indices, test_labels, train_val_labels)
    return {
        'index': index,
        'features_gallery_norm': features_gallery_norm,
        'features_queries_norm': features_queries_norm,
        'test_labels': test_labels,
        'train_val_labels': train_val_labels,
        'label_encoder': label_encoder,
        'metrics': {'nDCG@5': avg_ndcg, 'Recall@5': avg_recall, 'mAP@10': avg_map},
        'metadata_path': metadata_path
    }


def initialize_model(model_type='vit', model_path=None):
    if model_type != 'vit':
        raise ValueError("Only 'vit' model_type is supported.")
    if model_path is None:
        model_path = MODEL_PATH
    model = models.vit_b_16(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.heads.head = torch.nn.Identity()
    return model

# if __name__ == "__main__":
#     base_path = '/mnt/miaai/STUDIES/his_img_GNN_classification/datasets/bach'
#     results = run_full_pipeline(base_path)

