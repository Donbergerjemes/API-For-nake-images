from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import clip
from PIL import Image
import io

app = FastAPI()

# CORS pour autoriser le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger CLIP
print("🔄 Chargement du modèle CLIP...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"✅ Modèle chargé sur {device}")

# Catégories pour l'analyse
categories = ["safe for work", "not safe for work", "nudity", "sexual content"]
text_inputs = clip.tokenize(categories).to(device)

# 🔧 SEUIL - Ajuste cette valeur si besoin
# 0.5 = très strict (bloque presque tout)
# 0.75 = équilibré (recommandé)
# 0.85 = tolérant (seul l'explicite pur est bloqué)

SEUIL_NSFW = 0.75

@app.get("/")
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Lire l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Prétraiter
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Inférence
        with torch.no_grad():
            logits_per_image, _ = model(image_input, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        # Construire les résultats
        results = {categories[i]: float(probs[0][i]) for i in range(len(categories))}
        
        # Décision finale avec le seuil
        nsfw_score = results["not safe for work"] + results["nudity"] + results["sexual content"]
        is_nsfw = nsfw_score > SEUIL_NSFW
        
        return {
            "scores": results,
            "is_nsfw": is_nsfw,
            "confidence": float(max(probs[0]))
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)