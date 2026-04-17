from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
import numpy as np
import cv2
import base64
import os
import logging

from Encryption import (
    UNetKeyGenerator,
    ImageEncryptor,
    ImageDecryptor,
    calculate_entropy,
    calculate_npcr,
    calculate_uaci,
    ChaoticSeedGenerator,
)

# --- setup ---
ROOT_DIR = Path(__file__).parent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cybersecurity Image Encryption API")
api_router = APIRouter(prefix="/api")

# Globals
key_generator = None
encryptor = None
decryptor = None
model_ready = False

# Pydantic models
class EncryptionResponse(BaseModel):
    encrypted_image: str
    key_data: dict
    chaotic_params: list
    image_shape: list
    metrics: dict

class DecryptionRequest(BaseModel):
    encrypted_image: str
    chaotic_params: list
    perm_indices: list
    image_shape: list

# Startup event: initialize/load model
@app.on_event("startup")
async def startup_event():
    global key_generator, encryptor, decryptor, model_ready
    logger.info("Initializing U-Net Key Generator...")

    model_path = ROOT_DIR / "unet_key_generator.keras"
    key_generator = UNetKeyGenerator(input_size=64, output_size=256)

    if model_path.exists():
        logger.info("Loading pre-trained model...")
        key_generator.load_model(str(model_path))
    else:
        logger.info("Training new U-Net model (quick tiny training for demo)...")
        # small synthetic training to produce something; for real usage train properly
        chaotic_seeds = []
        target_keys = []
        for _ in range(8):
            x0 = np.random.uniform(0.01, 0.99)
            y0 = np.random.uniform(0.01, 0.99)
            r0 = np.random.uniform(0.01, 0.99)
            seed = ChaoticSeedGenerator.create_chaotic_seed_image(x0, y0, r0, size=64)
            chaotic_seeds.append(seed[..., np.newaxis])
            # generate random target as demonstration
            t = np.random.rand(256, 256).astype(np.float32) * 2 - 1
            target_keys.append(t[..., np.newaxis])
        chaotic_seeds = np.array(chaotic_seeds, dtype=np.float32)
        target_keys = np.array(target_keys, dtype=np.float32)
        key_generator.train(chaotic_seeds, target_keys, epochs=2, batch_size=2)
        key_generator.save_model(str(model_path))
        logger.info("Model trained and saved.")

    encryptor = ImageEncryptor(key_generator)
    decryptor = ImageDecryptor(key_generator)
    model_ready = True
    logger.info("Encryption engine ready.")

# Routes
@api_router.get("/")
async def root():
    return {"message": "Image Encryption API running successfully"}

@api_router.get("/model-status")
async def get_model_status():
    return {"ready": model_ready}

@api_router.post("/encrypt", response_model=EncryptionResponse)
async def encrypt_image(file: UploadFile = File(...)):
    global encryptor, model_ready
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model is initializing")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        original_shape = image.shape
        image = cv2.resize(image, (256, 256))

        chaotic_params = (
            float(np.random.uniform(0.1, 0.9)),
            float(np.random.uniform(0.1, 0.9)),
            float(np.random.uniform(0.1, 0.9)),
        )

        encrypted, key_matrix, perm_indices, params = encryptor.encrypt(image, chaotic_params)

        orig_entropy = calculate_entropy(image)
        enc_entropy = calculate_entropy(encrypted)
        npcr = calculate_npcr(image, encrypted)
        uaci = calculate_uaci(image, encrypted)

        _, buffer = cv2.imencode(".png", encrypted)
        encrypted_b64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "encrypted_image": encrypted_b64,  # plain base64 string (no data URI prefix)
            "key_data": {"perm_indices": perm_indices.tolist()},
            "chaotic_params": list(params),
            "image_shape": list(original_shape),
            "metrics": {
                "original_entropy": float(orig_entropy),
                "encrypted_entropy": float(enc_entropy),
                "npcr": float(npcr),
                "uaci": float(uaci),
            },
        }

    except Exception as e:
        logger.exception("Encryption failed")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/decrypt")
async def decrypt_image(request: DecryptionRequest):
    global decryptor, model_ready
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model is initializing")

    try:
        encrypted_bytes = base64.b64decode(request.encrypted_image)
        nparr = np.frombuffer(encrypted_bytes, np.uint8)
        encrypted_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if encrypted_image is None:
            raise HTTPException(status_code=400, detail="Invalid encrypted image")

        perm_indices = np.array(request.perm_indices, dtype=np.int64)
        chaotic_params = tuple(request.chaotic_params)

        decrypted = decryptor.decrypt(encrypted_image, chaotic_params, perm_indices)
        _, buffer = cv2.imencode(".png", decrypted)
        decrypted_b64 = base64.b64encode(buffer).decode("utf-8")

        return {"decrypted_image": decrypted_b64, "message": "Decryption successful"}

    except Exception as e:
        logger.exception("Decryption failed")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/download_encrypted")
async def download_encrypted(payload: dict):
    try:
        img_b64 = payload.get("encrypted_image")
        if not img_b64:
            raise HTTPException(status_code=400, detail="No encrypted_image provided")
        # remove data URI prefix if present
        if isinstance(img_b64, str) and img_b64.startswith("data:image"):
            img_b64 = img_b64.split(",")[1]
        img_bytes = base64.b64decode(img_b64)
        return Response(content=img_bytes, media_type="image/png",
                        headers={"Content-Disposition": "attachment; filename=encrypted_image.png"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Frontend serving (static)
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()
    
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)