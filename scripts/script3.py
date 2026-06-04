import numpy as np 
import pandas as pd
import torch
import open_clip
import os
from urllib.request import urlopen
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

### OneCycleLR y solo una capa descongelada

device = "cuda" if torch.cuda.is_available() else "cpu"

#Ponemos la ruta al dataset desde kaggle
base_path = "/home/ashley-bravo/otro/roco/all_data/"
train_path = os.path.join(base_path, "train")
img_path = os.path.join(train_path, "radiology", "images")
csv_path = os.path.join(base_path, "Trainingdataset.csv")

df = pd.read_csv(csv_path)

df["image_path"] = df["name"].apply(lambda x: f"{img_path}/{x}")
print(df.head())

#Ponemos la ruta al dataset desde kaggle
val_base_path = "/home/ashley-bravo/otro/roco/all_data/"
val_path = os.path.join(val_base_path, "validation")
val_img = os.path.join(val_path, "radiology", "images")
val_csv = os.path.join(val_path, "radiology", "valdata.csv")

df_val = pd.read_csv(val_csv)
df_val["image_path"] = df_val["name"].apply(lambda x: os.path.join(val_img, x))

# Filtrar existentes y válidas (igual que en train)
real_val   = set(os.listdir(val_img))
df_val     = df_val[df_val["name"].isin(real_val)].reset_index(drop=True)

valid_val  = []
for name in tqdm(df_val["name"], desc="Verificando val"):
    path = os.path.join(val_img, name)
    try:
        with Image.open(path) as img:
            img.verify()
        valid_val.append(name)
    except Exception:
        pass

df_val = df_val[df_val["name"].isin(set(valid_val))].reset_index(drop=True)
print(f"Número de imágenes para validación: {len(df_val)} ")

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

model = model.to(device)

class ValDataset(Dataset):
    def __init__(self, df, preprocess, tokenizer):
        self.df         = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.tokenizer  = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.preprocess(image)
        text  = self.tokenizer(row["caption"], context_length=256)[0]
        return image, text

def extract_embeddings(model, val_loader, device):
    model.eval()
    all_image_features = []
    all_text_features  = []

    with torch.no_grad():
        for images, texts in tqdm(val_loader, desc="Extrayendo embeddings"):
            images = images.to(device)
            texts  = texts.to(device)

            img_feat, txt_feat, _ = model(images, texts)

            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            all_image_features.append(img_feat.cpu())
            all_text_features.append(txt_feat.cpu())

    return torch.cat(all_image_features), torch.cat(all_text_features)

class TrainDataset(Dataset):
    def __init__(self, df, preprocess, tokenizer):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        image = self.preprocess(image)

        text = self.tokenizer(
            row["caption"],
            context_length=256
        )[0]

        return image, text
    
train_dataset = TrainDataset(df, preprocess_train, tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_dataset_base = ValDataset(df_val, preprocess_val, tokenizer)
val_loader_ft  = DataLoader(val_dataset_base, batch_size=64,
                               shuffle=False, num_workers=2, pin_memory=True)

for param in model.parameters():
    param.requires_grad = False

#Solo entrenamos dos ultima capas
for param in model.visual.trunk.blocks[-2:].parameters():
    param.requires_grad = True

# Verificamos cuántos parámetros se entrenan
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parámetros totales:    {total:,}")
print(f"Parámetros entrenables: {trainable:,}  ({100*trainable/total:.1f}%)")

epochs = 5

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-7,
    weight_decay=1e-4
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=epochs,      # un ciclo completo en N épocas
    eta_min=1e-8       # nunca baja de esto
)

def recall_at_k(image_features, text_features, ks=(1, 5, 10)):
    """
    Para cada imagen, busca su caption correcto entre los K más similares.
    El ground truth es la diagonal (imagen i ↔ texto i).
    """
    # Matriz de similitud (N x N)
    sim_matrix = image_features @ text_features.T  # cosine sim (ya normalizados)

    results = {}
    N = sim_matrix.shape[0]

    for k in ks:
        # Image → Text: para cada imagen, top-K textos más similares
        topk_i2t = sim_matrix.topk(k, dim=1).indices        # (N, k)
        correct_i2t = (topk_i2t == torch.arange(N).unsqueeze(1)).any(dim=1)
        r_i2t = correct_i2t.float().mean().item() * 100

        # Text → Image: para cada texto, top-K imágenes más similares
        topk_t2i = sim_matrix.T.topk(k, dim=1).indices      # (N, k)
        correct_t2i = (topk_t2i == torch.arange(N).unsqueeze(1)).any(dim=1)
        r_t2i = correct_t2i.float().mean().item() * 100

        results[f"R@{k} I→T"] = round(r_i2t, 2)
        results[f"R@{k} T→I"] = round(r_t2i, 2)

    return results

print("Iniciaremos entrenamiento")
# ─── Loop con early stopping ───────────────────────────────────────
best_recall = 0
no_improve  = 0
patience    = 3
train_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_idx, (images, texts) in enumerate(train_loader):
        images = images.to(device)
        texts  = texts.to(device)

        image_features, text_features, logit_scale = model(images, texts)

        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text  = logits_per_image.T
        labels = torch.arange(len(images)).to(device)

        loss = (F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text,  labels)) / 2

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        total_loss += loss.item()

    # scheduler paso por época (no por batch)
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f} | "
          f"lr: {current_lr:.2e}")

    # ── Evaluar cada época ─────────────────────────────────────────
    img_feat, txt_feat = extract_embeddings(model, val_loader_ft, device)
    metrics = recall_at_k(img_feat, txt_feat, ks=(1,))
    current_recall = metrics["R@1 I→T"]

    print(f"         R@1 I→T: {current_recall:.2f}%  (mejor: {best_recall:.2f}%)")

    if current_recall > best_recall:
        best_recall = current_recall
        no_improve  = 0
        torch.save(model.state_dict(), "biomedclip_best.pt")
        print(f"         ✓ Nuevo mejor guardado")
    else:
        no_improve += 1
        print(f"         Sin mejora ({no_improve}/{patience})")
        if no_improve >= patience:
            print(f"\nEarly stopping en época {epoch+1}")
            break

print(f"\nMejor R@1 I→T alcanzado: {best_recall:.2f}%")

# ─── 7. Guardar ───────────────────────────────────────────────────
torch.save(model.state_dict(), "biomedclip_roco_finetuned_5_3.pt")
print("Modelo guardado.")
