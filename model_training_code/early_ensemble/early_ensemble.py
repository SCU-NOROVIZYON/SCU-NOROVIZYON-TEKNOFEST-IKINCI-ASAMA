<<<<<<< HEAD
import os
import torch
import torch.nn as nn
import timm
import pydicom
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Backbone modeli: ConvNeXt, sınıflandırma katmanı yok, sadece özellik çıkarıcı
class CustomConvNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("convnext_base", pretrained=False, num_classes=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.global_avg_pool(x)
        return x.view(x.size(0), -1)

# --- MLP sınıflandırıcı
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

# --- Dataset: vaka bazlı ve modalitelere göre DICOM dosyalarını yükler
class VakaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for vaka in os.listdir(cls_path):
                vaka_path = os.path.join(cls_path, vaka)
                if not os.path.isdir(vaka_path):
                    continue
                modalite_dict = {}
                for mod in ['adc','dwi','t2a','ct']:
                    mod_path = os.path.join(vaka_path, mod)
                    if os.path.isdir(mod_path):
                        dcm_files = [os.path.join(mod_path, f) for f in os.listdir(mod_path) if f.lower().endswith('.dcm')]
                        if dcm_files:
                            modalite_dict[mod] = dcm_files
                if modalite_dict:
                    self.samples.append((modalite_dict, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        modalite_dict, label = self.samples[idx]
        images_per_modality = {}
        for mod, paths in modalite_dict.items():
            imgs = []
            for p in paths:
                ds = pydicom.dcmread(p)
                img = ds.pixel_array.astype(np.float32)
                img -= img.min()
                img /= (img.max() + 1e-5)
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)
                img = Image.fromarray((img*255).astype(np.uint8))
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
            images_per_modality[mod] = torch.stack(imgs)
        return images_per_modality, label

# --- Collate function: batch içindeki modalite dictlerini ve labelleri ayrıştırır
def collate_fn(batch):
    modalite_dicts = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return modalite_dicts, labels

# --- Özellik çıkarma fonksiyonu: her vaka için modaliteler bazında özellikler çıkarılır
def extract_features(dataloader, models):
    features, labels = [], []
    for modalite_dicts, label_batch in tqdm(dataloader, desc="Extracting test features"):
        for modalite_dict, label in zip(modalite_dicts, label_batch):
            vaka_feats = []
            for mod in ['adc','dwi','t2a','ct']:
                if mod in modalite_dict:
                    imgs = modalite_dict[mod].to(device)
                    with torch.no_grad():
                        feat = models[mod](imgs)
                        avg_feat = feat.mean(dim=0)
                    vaka_feats.append(avg_feat.cpu())
                else:
                    feat_dim = list(models.values())[0].backbone.num_features
                    vaka_feats.append(torch.zeros(feat_dim))
            features.append(torch.cat(vaka_feats))
            labels.append(label)
    return torch.stack(features), torch.tensor(labels)

# --- Önceden eğitilmiş özellik çıkarıcı modelleri yükleme
def load_feature_extractors():
    modalities = ['adc','dwi','t2a','ct']
    feature_extractors = {}
    for mod in modalities:
        model = CustomConvNeXt().to(device)
        model.load_state_dict(torch.load(fr"YOUR_PATH_HERE\{mod}\{mod}_gap_focal_best_model.pth", map_location=device), strict=False)
        model.eval()
        feature_extractors[mod] = model
    return feature_extractors

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Test veri seti
    test_dataset = VakaDataset(r"YOUR_PATH_HERE\test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Özellik çıkarıcı modelleri yükle
    feature_extractors = load_feature_extractors()
    print("Extracting test features...")
    test_features, test_labels = extract_features(test_loader, feature_extractors)

    # MLP modelini yükle
    mlp_input_dim = test_features.shape[1]
    num_classes = len(test_dataset.classes)
    mlp_model = MLP(mlp_input_dim, num_classes).to(device)
    mlp_model.load_state_dict(torch.load("mlp_model.pth", map_location=device))
    mlp_model.eval()

    # Tahmin yap
    with torch.no_grad():
        outputs = mlp_model(test_features.to(device))
        preds = torch.argmax(outputs, dim=1).cpu()

    # Sonuçları yazdır
    print("Classification Report:")
    print(classification_report(test_labels, preds, target_names=test_dataset.classes,  digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, preds))
=======
import os
import torch
import torch.nn as nn
import timm
import pydicom
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Backbone modeli: ConvNeXt, sınıflandırma katmanı yok, sadece özellik çıkarıcı
class CustomConvNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("convnext_base", pretrained=False, num_classes=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.global_avg_pool(x)
        return x.view(x.size(0), -1)

# --- MLP sınıflandırıcı
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

# --- Dataset: vaka bazlı ve modalitelere göre DICOM dosyalarını yükler
class VakaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for vaka in os.listdir(cls_path):
                vaka_path = os.path.join(cls_path, vaka)
                if not os.path.isdir(vaka_path):
                    continue
                modalite_dict = {}
                for mod in ['adc','dwi','t2a','ct']:
                    mod_path = os.path.join(vaka_path, mod)
                    if os.path.isdir(mod_path):
                        dcm_files = [os.path.join(mod_path, f) for f in os.listdir(mod_path) if f.lower().endswith('.dcm')]
                        if dcm_files:
                            modalite_dict[mod] = dcm_files
                if modalite_dict:
                    self.samples.append((modalite_dict, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        modalite_dict, label = self.samples[idx]
        images_per_modality = {}
        for mod, paths in modalite_dict.items():
            imgs = []
            for p in paths:
                ds = pydicom.dcmread(p)
                img = ds.pixel_array.astype(np.float32)
                img -= img.min()
                img /= (img.max() + 1e-5)
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)
                img = Image.fromarray((img*255).astype(np.uint8))
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
            images_per_modality[mod] = torch.stack(imgs)
        return images_per_modality, label

# --- Collate function: batch içindeki modalite dictlerini ve labelleri ayrıştırır
def collate_fn(batch):
    modalite_dicts = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return modalite_dicts, labels

# --- Özellik çıkarma fonksiyonu: her vaka için modaliteler bazında özellikler çıkarılır
def extract_features(dataloader, models):
    features, labels = [], []
    for modalite_dicts, label_batch in tqdm(dataloader, desc="Extracting test features"):
        for modalite_dict, label in zip(modalite_dicts, label_batch):
            vaka_feats = []
            for mod in ['adc','dwi','t2a','ct']:
                if mod in modalite_dict:
                    imgs = modalite_dict[mod].to(device)
                    with torch.no_grad():
                        feat = models[mod](imgs)
                        avg_feat = feat.mean(dim=0)
                    vaka_feats.append(avg_feat.cpu())
                else:
                    feat_dim = list(models.values())[0].backbone.num_features
                    vaka_feats.append(torch.zeros(feat_dim))
            features.append(torch.cat(vaka_feats))
            labels.append(label)
    return torch.stack(features), torch.tensor(labels)

# --- Önceden eğitilmiş özellik çıkarıcı modelleri yükleme
def load_feature_extractors():
    modalities = ['adc','dwi','t2a','ct']
    feature_extractors = {}
    for mod in modalities:
        model = CustomConvNeXt().to(device)
        model.load_state_dict(torch.load(fr"YOUR_PATH_HERE\{mod}\{mod}_gap_focal_best_model.pth", map_location=device), strict=False)
        model.eval()
        feature_extractors[mod] = model
    return feature_extractors

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Test veri seti
    test_dataset = VakaDataset(r"YOUR_PATH_HERE\test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Özellik çıkarıcı modelleri yükle
    feature_extractors = load_feature_extractors()
    print("Extracting test features...")
    test_features, test_labels = extract_features(test_loader, feature_extractors)

    # MLP modelini yükle
    mlp_input_dim = test_features.shape[1]
    num_classes = len(test_dataset.classes)
    mlp_model = MLP(mlp_input_dim, num_classes).to(device)
    mlp_model.load_state_dict(torch.load("mlp_model.pth", map_location=device))
    mlp_model.eval()

    # Tahmin yap
    with torch.no_grad():
        outputs = mlp_model(test_features.to(device))
        preds = torch.argmax(outputs, dim=1).cpu()

    # Sonuçları yazdır
    print("Classification Report:")
    print(classification_report(test_labels, preds, target_names=test_dataset.classes,  digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, preds))
>>>>>>> bbb7611177ae268e5489c880ecb9f1d964437d2c
