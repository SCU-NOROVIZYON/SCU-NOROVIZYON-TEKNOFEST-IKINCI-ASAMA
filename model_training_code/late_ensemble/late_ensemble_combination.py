import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pydicom
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomConvNeXt(nn.Module):
    """
    ConvNeXt tabanlı model, sınıflandırma için.
    """
    def __init__(self, num_classes):
        super().__init__()
        import timm
        self.backbone = timm.create_model("convnext_base", pretrained=True, num_classes=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class EnsembleTestDatasetVakaBazli(Dataset):
    """
    Vaka bazlı test dataset'i, farklı modalitelerdeki DICOM görüntüleri yükler.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = []

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            vaka_list = os.listdir(cls_dir)
            for vaka in vaka_list:
                vaka_dir = os.path.join(cls_dir, vaka)
                if not os.path.isdir(vaka_dir):
                    continue
                modal_paths = {}
                for modal in ['adc', 'dwi', 'ct', 't2a']:
                    modal_dir = os.path.join(vaka_dir, modal)
                    if os.path.isdir(modal_dir):
                        dcm_files = [f for f in os.listdir(modal_dir) if f.lower().endswith('.dcm')]
                        if dcm_files:
                            full_paths = [os.path.join(modal_dir, f) for f in dcm_files]
                            modal_paths[modal] = full_paths
                if modal_paths:
                    self.samples.append((vaka, self.class_to_idx[cls], modal_paths))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vaka_id, label, modal_paths = self.samples[idx]
        images = {}

        for modal in ['adc', 'dwi', 'ct', 't2a']:
            images[modal] = []
            if modal in modal_paths:
                for dcm_path in modal_paths[modal]:
                    ds = pydicom.dcmread(dcm_path)
                    img = ds.pixel_array.astype(np.float32)
                    img -= img.min()
                    img /= (img.max() + 1e-5)
                    img = np.stack([img] * 3, axis=-1)
                    pil_img = Image.fromarray((img * 255).astype(np.uint8))
                    if self.transform:
                        pil_img = self.transform(pil_img)
                    images[modal].append(pil_img)
        return images, label, vaka_id


# Transform işlemleri
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Sınıf sayısı (gerekirse değiştirin)
num_classes = 3

# Model ağırlıklarının yolunu kendi sisteminize göre ayarlayın
model_paths = {
    'adc': "path/to/models/adc_gap_focal_best_model.pth",
    't2a': "path/to/models/t2a_gap_focal_best_model.pth",
    'dwi': "path/to/models/dwi_gap_focal_best_model.pth",
    'ct': "path/to/models/ct_gap_focal_best_model.pth",
}

# Modelleri yükle
all_models = {}
for modal, path in model_paths.items():
    model = CustomConvNeXt(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    all_models[modal] = model


def ensemble_predict(images_dict, selected_modalities, selected_models):
    """
    Modalitelere göre tahmin yapar, tahminleri ortalama ile ensemble eder.
    """
    with torch.no_grad():
        modal_preds = []
        for modal, model in zip(selected_modalities, selected_models):
            imgs = images_dict[modal]
            if len(imgs) == 0:
                modal_preds.append(np.zeros(num_classes))
                continue

            preds_modal = []
            for img in imgs:
                img = img.squeeze()
                if len(img.shape) != 3:
                    raise ValueError(f"Expected 3D tensor after squeeze, got shape {img.shape}")
                x = img.unsqueeze(0).to(device)
                output = model(x)
                probs = torch.softmax(output, dim=1)
                preds_modal.append(probs.cpu().numpy()[0])

            modal_mean = np.mean(preds_modal, axis=0)
            modal_preds.append(modal_mean)

        preds_mean = np.mean(modal_preds, axis=0)
        pred_label = np.argmax(preds_mean)
        return pred_label, preds_mean


if __name__ == "__main__":
    # Test veri dizini (kendi yolunuza göre değiştirin)
    test_root = "path/to/test"

    test_dataset = EnsembleTestDatasetVakaBazli(test_root, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    modalities = ['adc', 'dwi', 'ct', 't2a']

    # Farklı modalite kombinasyonlarıyla test et
    for r in [2, 3, 4]:
        print(f"\n=== {r}'li Kombinasyonlar ===\n")
        for combo in combinations(modalities, r):
            selected_modalities = list(combo)
            selected_models = [all_models[mod] for mod in selected_modalities]

            all_true = []
            all_pred = []
            all_vaka = []

            for images, label, vaka_id in tqdm(test_loader, desc=f"Kombinasyon: {selected_modalities}"):
                pred_label, _ = ensemble_predict(images, selected_modalities, selected_models)
                all_true.append(label.item())
                all_pred.append(pred_label)
                all_vaka.append(vaka_id[0])

            print(f"\n--- Kombinasyon: {selected_modalities} ---")
            print(classification_report(all_true, all_pred, target_names=test_dataset.classes, digits=4))

            cm = confusion_matrix(all_true, all_pred)
            print("Confusion Matrix:")
            print("Gerçek Sınıflar \\ Tahmin Edilen Sınıflar")
            print("  " + "  ".join(f"{cls:>8}" for cls in test_dataset.classes))
            for i, row in enumerate(cm):
                row_str = "  ".join(f"{val:8}" for val in row)
                print(f"{test_dataset.classes[i]:<15} {row_str}")