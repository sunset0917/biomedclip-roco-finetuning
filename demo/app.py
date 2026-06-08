import torch
import os
import pandas as pd
from tqdm import tqdm
import open_clip
import gradio as gr
from PIL import Image
from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="lulu12lemon/ROCO_test",
    repo_type="dataset",
    local_dir="ROCO_test"
)

CSV_PATH = "radiologytestdata_clean.csv"
IMAGE_DIR = "ROCO_test/radiology/images"
MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(CSV_PATH)
image_paths = df["image_path"].tolist()
captions = df["caption"].fillna("").tolist()

model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
model = model.to(device)
model.eval()

def greet(name):
    return "Hello " + name + "!!"

def encode_text(text):
    with torch.no_grad():
        tokens = tokenizer([text]).to(device)

        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(
            dim=-1,
            keepdim=True
        )

    return text_features

def encode_image(image):
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1,keepdim=True)

    return image_features

def text_to_image(query):

    text_features = encode_text(query)
    sims = text_features @ image_features_db.T
    topk = torch.topk(sims.squeeze(),k=5).indices.cpu().numpy()
    return [image_paths[i] for i in topk]

def image_to_text(image):

    image_features = encode_image(image)
    sims = image_features @ caption_features_db.T
    topk = torch.topk(sims.squeeze(),k=5).indices.cpu().numpy()
    return "\n".join([captions[i] for i in topk])


image_features_db = []

print("Encoding images...")

for path in tqdm(image_paths):
    try:
        img = Image.open(path).convert("RGB")
        feat = encode_image(img)
        image_features_db.append(feat.cpu())
    except Exception as e:
        print(f"Error: {path}")
        print(e)
        image_features_db.append(torch.zeros((1, 512)))

image_features_db = torch.cat(image_features_db, dim=0).to(device)

print("Image database ready")

caption_features_db = []

print("Encoding captions...")

for caption in tqdm(captions):
    feat = encode_text(caption)
    caption_features_db.append(feat.cpu())
caption_features_db = torch.cat(caption_features_db, dim=0).to(device)

print("Caption database ready")

with gr.Blocks() as demo:
    gr.Markdown("BiomedCLIP Retrieval")
    with gr.Tab("Text → Image"):
        text_input = gr.Textbox(label="Medical Query")
        gallery = gr.Gallery(label="Top Results",columns=5)
        btn = gr.Button("Search")
        btn.click(fn=text_to_image,inputs=text_input,outputs=gallery)

    with gr.Tab("Image → Text"):
        img_input = gr.Image(type="pil")
        text_output = gr.Textbox(label="Retrieved Captions",lines=10)
        btn2 = gr.Button("Search")
        btn2.click(fn=image_to_text,inputs=img_input,outputs=text_output)

demo.launch()
