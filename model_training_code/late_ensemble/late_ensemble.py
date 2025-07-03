import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pydicom
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Tanımı ---
class CustomConvNeXt(nn.Module):
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

# --- Görüntü dönüşümleri ---
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Dataset Sınıfı ---
class EnsembleTestDatasetVakaBazli(Dataset):
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

# --- Model ağırlıkları dosya yolları ---
num_classes = 3
model_paths = {
    'adc': r"YOUR_PATH_HERE/adc/adc_gap_focal_best_model.pth",
    't2a': r"YOUR_PATH_HERE/t2a/t2a_gap_focal_best_model.pth",
    'dwi': r"YOUR_PATH_HERE/dwi/dwi_gap_focal_best_model.pth",
    'ct': r"YOUR_PATH_HERE/ct/ct_gap_focal_best_model.pth",
}

# --- Modelleri yükleme ---
model_adc = CustomConvNeXt(num_classes=num_classes).to(device)
model_adc.load_state_dict(torch.load(model_paths['adc'], map_location=device))
model_adc.eval()

model_dwi = CustomConvNeXt(num_classes=num_classes).to(device)
model_dwi.load_state_dict(torch.load(model_paths['dwi'], map_location=device))
model_dwi.eval()

model_ct = CustomConvNeXt(num_classes=num_classes).to(device)
model_ct.load_state_dict(torch.load(model_paths['ct'], map_location=device))
model_ct.eval()

model_t2a = CustomConvNeXt(num_classes=num_classes).to(device)
model_t2a.load_state_dict(torch.load(model_paths['t2a'], map_location=device))
model_t2a.eval()

# --- Ensemble tahmin fonksiyonu ---
def ensemble_predict(images_dict):
    with torch.no_grad():
        modal_preds = []
        for modal, model in zip(['adc', 'dwi', 'ct', 't2a'], [model_adc, model_dwi, model_ct, model_t2a]):
            imgs = images_dict.get(modal, [])
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

# --- Test veri yolu ---
test_root = r"YOUR_PATH_HERE/TrainTest/test"

# --- Dataset ve DataLoader ---
test_dataset = EnsembleTestDatasetVakaBazli(test_root, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# --- Tahmin ve Değerlendirme ---
all_true = []
all_pred = []
all_vaka = []

for images, label, vaka_id in tqdm(test_loader, desc="Ensemble Testing"):
    pred_label, _ = ensemble_predict(images)
    all_true.append(label.item())
    all_pred.append(pred_label)
    all_vaka.append(vaka_id[0])

print(classification_report(all_true, all_pred, target_names=test_dataset.classes, digits=4))

cm = confusion_matrix(all_true, all_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Ensemble Confusion Matrix')
plt.show()