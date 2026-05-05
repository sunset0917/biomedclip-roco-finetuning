import numpy as np 
import pandas as pd
import torch
import open_clip
import os
from urllib.request import urlopen
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

#Ponemos la ruta al dataset desde kaggle
base_path = "/home/ashley-bravo/otro/roco/all_data/"
train_path = os.path.join(base_path, "train")
img_path = os.path.join(train_path, "radiology", "images")
csv_path = os.path.join(base_path, "Trainingdataset.csv")

df = pd.read_csv(csv_path)

df["image_path"] = df["name"].apply(lambda x: f"{img_path}/{x}")
print(df.head())

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

model = model.to(device)

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

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=1e-4
)

epochs = 15
print("Iniciaremos entrenamiento")
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
        optimizer.step()

        total_loss += loss.item()


    print(f"\nEpoch {epoch+1} completada de /{epochs} — ")

# ─── 7. Guardar ───────────────────────────────────────────────────
torch.save(model.state_dict(), "biomedclip_roco_finetuned_15.pt")
print("Modelo guardado.")


