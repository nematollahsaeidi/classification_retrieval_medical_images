
import os, numpy as np, torch, time, logging, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, cohen_kappa_score
import tifffile
from tqdm import tqdm
from config import cfg

SEED = cfg["project"]["seed"]
DATASET_TYPE = cfg["dataset"]["name"]
BASE_PATH = cfg["paths"]["base_dataset"]
VIT_PATH = cfg["paths"]["pretrained_vit"]
NUM_CLASSES = cfg["dataset"]["num_classes"]
BATCH_SIZE = cfg["baseline"]["batch_size"]
EPOCHS = cfg["baseline"]["num_epochs"]
LR = cfg["baseline"]["learning_rate"]
K_FOLDS = cfg["training"]["k_folds"]
MODEL_DIR = cfg["paths"]["models_dir"]
MODEL_TYPE = cfg["feature_model"]["type"]


def setup_environment(seed=42, log_file=None, model_type='vit', dataset_type='BACH'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    path_model = f'{model_type}_{dataset_type}'
    if log_file is None:
        log_file = f'{cfg["paths"]["logs_dir"]}/{path_model}/out.log'
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


class MedicalDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        if self.transform:
            image = self.transform(torch.tensor(image))
        return image, label


def get_transforms():
    mean = cfg["transforms"]["train"]["normalize"]["mean"]
    std = cfg["transforms"]["train"]["normalize"]["std"]
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(tuple(cfg["transforms"]["train"]["resize"])),
        transforms.RandomHorizontalFlip() if cfg["baseline"]["augmentations"][
            "random_horizontal_flip"] else transforms.Lambda(lambda x: x),
        transforms.RandomVerticalFlip() if cfg["baseline"]["augmentations"][
            "random_vertical_flip"] else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(cfg["baseline"]["augmentations"]["random_rotation"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(tuple(cfg["transforms"]["val_test"]["resize"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return train_transform, val_transform


def initialize_model(model_type, num_classes, model_path=None):
    if model_type != 'vit':
        raise ValueError("Only 'vit' model_type is supported.")
    if model_path is None:
        model_path = VIT_PATH
    model = models.vit_b_16(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


def train_model(model, train_loader, val_loader, num_epochs=30, lr=1e-4, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    best_model_state = None
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    return best_model_state


def evaluate_model(model, test_loader, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model = model.to(device)
    test_preds, test_labels_list, test_probs = [], [], []
    test_start_time = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, 1).cpu().numpy()
            test_preds.extend(preds)
            test_labels_list.extend(labels.numpy())
            test_probs.extend(probs)
    test_time = time.time() - test_start_time
    len_test_dataset = len(test_loader.dataset)
    conf_matrix = confusion_matrix(test_labels_list, test_preds)
    test_kappa = cohen_kappa_score(test_labels_list, test_preds, weights='quadratic')
    test_f1 = f1_score(test_labels_list, test_preds, average='macro')
    test_auc = roc_auc_score(test_labels_list, test_probs, multi_class='ovr')
    return {
        'conf_matrix': conf_matrix,
        'kappa': test_kappa,
        'f1': test_f1,
        'auc': test_auc,
        'test_time_per_sample': test_time / len_test_dataset
    }


def get_data_loaders(train_images, train_labels, val_images, val_labels, test_images, test_labels, batch_size=32,
                     train_transform=None, val_transform=None):
    train_dataset = MedicalDataset(train_images, train_labels, train_transform)
    val_dataset = MedicalDataset(val_images, val_labels, val_transform)
    test_dataset = MedicalDataset(test_images, test_labels, val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    del train_dataset, val_dataset, test_dataset
    return train_loader, val_loader, test_loader


def run_kfold(train_val_images, train_val_labels, test_images, test_labels, num_classes, k_folds=5, batch_size=32,
              num_epochs=300, lr=1e-4, seed=42, model_type='vit', dataset_type='BACH', model_dir='best_models'):
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    all_conf_matrices = []
    test_results = {'kappa': [], 'f1': [], 'auc': [], 'train_times': [], 'test_times': []}
    train_transform, val_transform = get_transforms()
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_images, train_val_labels)):
        train_images = train_val_images[train_idx]
        train_labels = train_val_labels[train_idx]
        val_images = train_val_images[val_idx]
        val_labels = train_val_labels[val_idx]
        train_loader, val_loader, test_loader = get_data_loaders(
            train_images, train_labels, val_images, val_labels, test_images, test_labels,
            batch_size, train_transform, val_transform
        )
        del train_images, train_labels, val_images, val_labels
        model = initialize_model(model_type, num_classes)
        train_start_time = time.time()
        best_model_state = train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr)
        train_time = time.time() - train_start_time
        os.makedirs(model_dir, exist_ok=True)
        model_path = f'{model_dir}/{model_type}_{dataset_type}/fold_{fold + 1}.pth'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(best_model_state, model_path)
        model.load_state_dict(best_model_state)
        eval_results = evaluate_model(model, test_loader)
        eval_results['train_time'] = train_time
        all_conf_matrices.append(eval_results['conf_matrix'])
        for key in ['kappa', 'f1', 'auc', 'train_times', 'test_times']:
            if key == 'train_times':
                test_results[key].append(train_time)
            elif key == 'test_times':
                test_results[key].append(eval_results['test_time_per_sample'])
            else:
                test_results[key].append(eval_results[key])
        del train_loader, val_loader, test_loader
    model = initialize_model(model_type, num_classes)
    num_params = sum(p.numel() for p in model.parameters())
    avg_test_kappa = np.mean(test_results['kappa'])
    std_test_kappa = np.std(test_results['kappa'])
    avg_test_f1 = np.mean(test_results['f1'])
    std_test_f1 = np.std(test_results['f1'])
    avg_test_auc = np.mean(test_results['auc'])
    std_test_auc = np.std(test_results['auc'])
    avg_train_time = np.mean(test_results['train_times'])
    avg_test_time = np.mean(test_results['test_times'])
    std_train_time = np.std(test_results['train_times'])
    std_test_time = np.std(test_results['test_times'])
    sum_conf_matrix = np.sum(all_conf_matrices, axis=0)
    return {
        'metrics': {
            'kappa': (avg_test_kappa, std_test_kappa),
            'f1': (avg_test_f1, std_test_f1),
            'auc': (avg_test_auc, std_test_auc),
            'train_time': (avg_train_time, std_train_time),
            'test_time': (avg_test_time, std_test_time)
        },
        'conf_matrix': sum_conf_matrix,
        'num_params': num_params
    }


def run_full_pipeline(base_path, seed=42, k_folds=5, batch_size=32, num_epochs=300, lr=1e-4, model_type='vit',
                      dataset_type='BACH', model_dir='best_models'):
    device, logger = setup_environment(seed, model_type=model_type, dataset_type=dataset_type)
    image_paths, labels = load_dataset(base_path, dataset_type)
    images = preprocess_images(image_paths)
    train_val_images, test_images, train_val_labels, test_labels, label_encoder, num_classes = split_dataset(images,
                                                                                                             labels,
                                                                                                             seed=seed)
    results = run_kfold(
        train_val_images, train_val_labels, test_images, test_labels, num_classes,
        k_folds, batch_size, num_epochs, lr, seed, model_type, dataset_type, model_dir
    )
    return results

# if __name__ == "__main__":
#    base_path = '/mnt/miaai/STUDIES/his_img_GNN_classification/datasets/bach'
#    results = run_full_pipeline(base_path)
#    print("Training completed. Results:", results)
