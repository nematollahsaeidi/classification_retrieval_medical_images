
import os, numpy as np, torch, random, time, tifffile, torch.nn as nn, logging
from tqdm import tqdm
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphDataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, \
    cohen_kappa_score
from scipy.cluster.vq import kmeans2
from scipy.ndimage import binary_dilation
from config import cfg

SEED = cfg["project"]["seed"]
DATASET_TYPE = cfg["dataset"]["name"]
BASE_PATH = cfg["paths"]["base_dataset"]
VIT_PATH = cfg["paths"]["pretrained_vit"]
NUM_CLUSTERS = cfg["graph"]["num_clusters"]
NUM_CLASSES = cfg["dataset"]["num_classes"]
BATCH_SIZE = cfg["training"]["batch_size"]
EPOCHS = cfg["training"]["num_epochs"]
LR = cfg["training"]["learning_rate"]
K_FOLDS = cfg["training"]["k_folds"]
MODEL_DIR = cfg["paths"]["models_dir"]
MODEL_TYPE = cfg["feature_model"]["type"]

os.environ["TRANSFORMERS_CACHE"] = cfg["env"]["TRANSFORMERS_CACHE"]
os.environ["HF_HOME"] = cfg["env"]["HF_HOME"]
os.environ["HF_DATASETS_CACHE"] = cfg["env"]["HF_DATASETS_CACHE"]


def setup_environment(**kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(SEED);
    np.random.seed(SEED);
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    log_file = f"{cfg['paths']['logs_dir']}/graph_{MODEL_TYPE}_{DATASET_TYPE}/{NUM_CLUSTERS}_cluster.log"
    logging.basicConfig(level=getattr(logging, cfg["logging"]["level"]),
                        format=cfg["logging"]["format"],
                        handlers=[logging.FileHandler(log_file, mode=cfg["logging"]["file_mode"]),
                                  logging.StreamHandler() if cfg["logging"]["stream"] else None])
    return device, logging.getLogger(__name__)


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
    for img_path in tqdm(image_paths):
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


def get_memory_size_mb(data):
    if isinstance(data, np.ndarray):
        return data.nbytes / (1024 * 1024)
    elif isinstance(data, torch.Tensor):
        return data.element_size() * data.nelement() / (1024 * 1024)
    return 0


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


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, model, transform, model_type='vit', num_clusters=8):
        self.device = next(model.parameters()).device
        self.model = model
        self.transform = transform
        self.images = images
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.graphs = []
        self.avg_graph_memory = []
        self.avg_build_graph_time = []
        build_graph_times = []
        graph_memory_sizes = []
        if num_clusters is None:
            num_clusters = 8
        for i in tqdm(range(len(images))):
            start_time = time.time()
            graph = self.build_graph_from_segmentation(images[i], num_clusters)
            build_graph_times.append(time.time() - start_time)
            if graph is not None and graph.num_nodes > 0:
                node_mem = get_memory_size_mb(graph.x)
                edge_mem = get_memory_size_mb(graph.edge_index)
                graph_memory_sizes.append(node_mem + edge_mem)
                self.graphs.append(graph)
            else:
                dummy_x = torch.zeros((1, 768), dtype=torch.float)
                dummy_edge_index = torch.empty((2, 0), dtype=torch.long)
                dummy = Data(x=dummy_x, edge_index=dummy_edge_index)
                self.graphs.append(dummy)
                graph_memory_sizes.append(0)
        self.avg_graph_memory = np.mean(graph_memory_sizes)
        self.avg_build_graph_time = np.mean(build_graph_times)

    def segment_image(self, img, num_clusters):
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        data = img.reshape((-1, 3)).astype(np.float32)
        labels, _ = kmeans2(data, num_clusters, minit='points')
        segmented = labels.reshape(img.shape[:2]).astype(int)
        return segmented

    def build_graph_from_segmentation(self, img, num_clusters):
        segmented = self.segment_image(img, num_clusters)
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
        for cx, cy in centroids:
            left = max(0, int(cx - crop_size // 2))
            top = max(0, int(cy - crop_size // 2))
            right = min(w, left + crop_size)
            bottom = min(h, top + crop_size)
            cropped = img[top:bottom, left:right]
            cropped_pil = TF.to_pil_image(cropped)
            resized_pil = TF.resize(cropped_pil, (crop_size, crop_size))
            tensor_crop = TF.to_tensor(resized_pil).unsqueeze(0).to(self.device)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            tensor_crop = normalize(tensor_crop)
            with torch.no_grad():
                emb = get_patch_embeddings(self.model, tensor_crop, model_type='vit', return_class_token=True)
            node_features_list.append(emb.squeeze(0).cpu().numpy())
        if len(node_features_list) == 0:
            return None
        x = torch.tensor(np.stack(node_features_list), dtype=torch.float)
        graph = Data(x=x, edge_index=edge_index)
        return graph

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


class GNN_GAT(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super(GNN_GAT, self).__init__()
        self.conv1 = GATConv(in_dim, 8, heads=heads, concat=True)
        self.conv2 = GATConv(8 * heads, 4, heads=heads, concat=True)
        self.fc = nn.Linear(4 * heads, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        return self.fc(x)


def train_gnn(model, train_loader, val_loader, num_epochs=300, lr=1e-4, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    best_val_loss = float('inf')
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(device)
                labels = labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    return best_model_state


def evaluate_gnn(model, test_loader, num_classes, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    test_preds, test_labels_list, test_probs = [], [], []
    start_time = time.time()
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels_list.extend(labels.numpy().flatten())
            test_probs.extend(probs)
    test_time = time.time() - start_time
    test_probs = np.array(test_probs)
    conf_matrix = confusion_matrix(test_labels_list, test_preds)
    kappa = cohen_kappa_score(test_labels_list, test_preds, weights='quadratic')
    acc = accuracy_score(test_labels_list, test_preds)
    f1 = f1_score(test_labels_list, test_preds, average='macro')
    balanced_acc = balanced_accuracy_score(test_labels_list, test_preds)
    if num_classes == 2:
        auc = roc_auc_score(test_labels_list, test_probs[:, 1])
    else:
        auc = roc_auc_score(test_labels_list, test_probs, multi_class='ovr')
    return {
        'conf_matrix': conf_matrix,
        'kappa': kappa,
        'acc': acc,
        'f1': f1,
        'balanced_acc': balanced_acc,
        'auc': auc,
        'test_time_per_sample': test_time / len(test_loader.dataset)
    }


def get_data_loaders(train_images, train_labels, val_images, val_labels, test_images, test_labels, batch_size=32,
                     feature_model=None, train_transform=None, val_transform=None, model_type='vit', num_clusters=None):
    train_dataset = GraphDataset(train_images, train_labels, feature_model, train_transform, model_type=model_type,
                                 num_clusters=num_clusters)
    val_dataset = GraphDataset(val_images, val_labels, feature_model, val_transform, model_type=model_type,
                               num_clusters=num_clusters)
    test_dataset = GraphDataset(test_images, test_labels, feature_model, val_transform, model_type=model_type,
                                num_clusters=num_clusters)
    train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    del train_dataset, val_dataset, test_dataset
    return train_loader, val_loader, test_loader


def get_transforms():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_test_transform


def initialize_feature_model(model_type='vit', model_path=None, device=None):
    if model_type != 'vit':
        raise ValueError("Only 'vit' model_type is supported.")
    if model_path is None:
        model_path = VIT_PATH
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.vit_b_16(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    in_dim = 768
    return model, in_dim


def run_kfold(train_val_images, train_val_labels, test_images, test_labels, num_classes, k_folds=5, batch_size=32,
              num_epochs=300, lr=1e-4, seed=42, model_type='vit', dataset_type='BACH', num_clusters=None,
              model_dir='best_models'):
    device, logger = setup_environment(seed, model_type=model_type, dataset_type=dataset_type,
                                       num_clusters=num_clusters)
    model_name = f'graph_{model_type}'
    feature_model, in_dim = initialize_feature_model(model_type, device=device)
    train_transform, val_transform = get_transforms()
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    all_conf_matrices = []
    test_results = {'kappa': [], 'f1': [], 'auc': [], 'acc': [], 'balanced_acc': []}
    train_times = []
    test_times = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_images, train_val_labels)):
        train_loader, val_loader, test_loader = get_data_loaders(
            train_val_images[train_idx], train_val_labels[train_idx],
            train_val_images[val_idx], train_val_labels[val_idx],
            test_images, test_labels,
            batch_size, feature_model, train_transform, val_transform, model_type, num_clusters
        )
        model = GNN_GAT(in_dim=in_dim, out_dim=num_classes)
        start_time = time.time()
        best_model_state = train_gnn(model, train_loader, val_loader, num_epochs, lr, device)
        train_time = time.time() - start_time
        train_times.append(train_time)
        os.makedirs(os.path.join(model_dir, f'{model_name}_{dataset_type}'), exist_ok=True)
        model_path = os.path.join(model_dir, f'{model_name}_{dataset_type}',
                                  f'fold_{fold + 1}_{num_clusters}_cluster.pth')
        torch.save(best_model_state, model_path)
        model.load_state_dict(best_model_state)
        eval_results = evaluate_gnn(model, test_loader, num_classes, device)
        test_times.append(eval_results['test_time_per_sample'])
        all_conf_matrices.append(eval_results['conf_matrix'])
        test_results['kappa'].append(eval_results['kappa'])
        test_results['acc'].append(eval_results['acc'])
        test_results['f1'].append(eval_results['f1'])
        test_results['balanced_acc'].append(eval_results['balanced_acc'])
        test_results['auc'].append(eval_results['auc'])
        del train_loader, val_loader, test_loader
    num_params = sum(p.numel() for p in model.parameters())
    avg_train_time = np.mean(train_times)
    std_train_time = np.std(train_times)
    avg_test_time = np.mean(test_times)
    std_test_time = np.std(test_times)
    avg_test_kappa = np.mean(test_results['kappa'])
    std_test_kappa = np.std(test_results['kappa'])
    avg_test_acc = np.mean(test_results['acc'])
    std_test_acc = np.std(test_results['acc'])
    avg_test_f1 = np.mean(test_results['f1'])
    std_test_f1 = np.std(test_results['f1'])
    avg_test_balanced_acc = np.mean(test_results['balanced_acc'])
    std_test_balanced_acc = np.std(test_results['balanced_acc'])
    avg_test_auc = np.mean(test_results['auc'])
    std_test_auc = np.std(test_results['auc'])
    sum_conf_matrix = np.sum(all_conf_matrices, axis=0)
    return {
        'metrics': {
            'kappa': (avg_test_kappa, std_test_kappa),
            'acc': (avg_test_acc, std_test_acc),
            'f1': (avg_test_f1, std_test_f1),
            'balanced_acc': (avg_test_balanced_acc, std_test_balanced_acc),
            'auc': (avg_test_auc, std_test_auc),
            'train_time': (avg_train_time, std_train_time),
            'test_time': (avg_test_time, std_test_time)
        },
        'conf_matrix': sum_conf_matrix,
        'num_params': num_params
    }


def run_full_pipeline(base_path, seed=42, k_folds=5, batch_size=32, num_epochs=300, lr=1e-4, model_type='vit',
                      dataset_type='BACH', num_clusters=None, model_dir='best_models'):
    image_paths, labels = load_dataset(base_path, dataset_type)
    images = preprocess_images(image_paths)
    train_val_images, test_images, train_val_labels, test_labels, label_encoder, num_classes = split_dataset(images,
                                                                                                             labels,
                                                                                                             seed=seed)
    results = run_kfold(
        train_val_images, train_val_labels, test_images, test_labels, num_classes,
        k_folds, batch_size, num_epochs, lr, seed, model_type, dataset_type, num_clusters, model_dir
    )
    return results

# if __name__ == "__main__":
#    base_path = '/mnt/miaai/STUDIES/his_img_GNN_classification/datasets/bach'
#    results = run_full_pipeline(base_path, num_clusters=None)
#    print("Training completed. Results:", results)

