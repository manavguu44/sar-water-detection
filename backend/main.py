from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

app = FastAPI()

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "SAR water detection backend is running"}

@app.get("/predict")
def predict(aoi: str = "delhi"):
    if aoi == "delhi":
        result_path = OUTPUT_DIR / "unet_full_result.png"

    elif aoi == "mumbai":
        result_path = OUTPUT_DIR / "mumbai_unet_result.png"

    else:
        return {"error": "AOI not supported"}

    if not result_path.exists():
        return {"error": f"{aoi} result not found"}

    return FileResponse(result_path)
@app.get("/comparison")
def comparison():
    result_path = OUTPUT_DIR / "model_comparison.png"

    if not result_path.exists():
        return {"error": "Comparison image not found. Run src/create_comparison.py first."}

    return FileResponse(result_path)
@app.get("/final-comparison")
def final_comparison(aoi: str = "delhi"):

    if aoi == "delhi":
        result_path = OUTPUT_DIR / "final_model_comparison.png"

    elif aoi == "mumbai":
        result_path = OUTPUT_DIR / "mumbai_model_comparison.png"

    else:
        return {"error": "AOI not supported"}

    if not result_path.exists():
        return {"error": f"{aoi} comparison not found"}

    return FileResponse(result_path)

@app.get("/overlay")
def overlay(aoi: str = "delhi"):
    if aoi == "delhi":
        result_path = OUTPUT_DIR / "delhi_water_overlay.png"
    elif aoi == "mumbai":
        result_path = OUTPUT_DIR / "mumbai_water_overlay.png"
    else:
        return {"error": "AOI not supported"}

    if not result_path.exists():
        return {"error": "Overlay not found. Run src/create_map_overlays.py first."}

    return FileResponse(result_path)


@app.get("/metrics")
def metrics(aoi: str = "delhi"):
    if aoi == "delhi":
        return {
            "aoi": "Delhi",
            "random_forest": {"f1": "0.9997", "iou": "0.9994"},
            "unet_vv_vh": {"f1": "0.9132", "iou": "0.8403"},
            "temporal_unet": {"f1": "0.8755", "iou": "0.7785"}
        }

    if aoi == "mumbai":
        return {
            "aoi": "Mumbai",
            "random_forest": {"f1": "0.9996", "iou": "0.9993"},
            "unet_vv_vh": {"f1": "0.9823", "iou": "0.9653"},
            "temporal_unet": {"f1": "0.9671", "iou": "0.9364"}
        }

    return {"error": "AOI not supported"}