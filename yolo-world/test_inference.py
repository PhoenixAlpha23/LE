import os
import sys
from pathlib import Path

# 1. Enforce strict offline mode BEFORE importing ultralytics
os.environ["YOLO_OFFLINE"] = "true"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from ultralytics import YOLOWorld


def run_offline_inference(
    model_path: str,
    source_path: str,
    output_dir: str,
    classes: list[str],
    conf: float = 0.25,
    iou: float = 0.5,
):
    # Verify local weights and input source exist before running
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Local model weights not found at: {model_path}")
    if not Path(source_path).exists():
        raise FileNotFoundError(f"Input image or directory not found at: {source_path}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 2. Load the locally downloaded model weights
    model = YOLOWorld(model_path)

    # 3. Set custom classes
    # Note: If your model already has custom classes baked into the .pt file,
    # skip calling set_classes() so it doesn't try to load the CLIP text encoder.
    if classes:
        model.set_classes(classes)

    # 4. Run batch inference on an image file OR an entire directory
    # stream=True prevents out-of-memory errors on large directories
    results = model.predict(
        source=source_path,
        conf=conf,
        iou=iou,
        save=True,               # Automatically saves annotated images
        project=str(out_path),   # Base output directory
        name="results",          # Subfolder inside output_dir
        exist_ok=True,           # Overwrite/append into the same folder without creating results2, results3...
        stream=True,             # Generator mode for memory safety on large folders
    )

    # Iterate through generator to execute inference and save files
    processed_count = 0
    for result in results:
        processed_count += 1
        print(f"[{processed_count}] Processed: {Path(result.path).name} -> Detections: {len(result.boxes)}")

    print(f"\nDone! Saved {processed_count} annotated image(s) to: {out_path / 'results'}")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    LOCAL_MODEL_PATH = "/media/aryan/E89E8AAD9E8A743A1/Github/LE/yolo-world/models/yolov8s-worldv2.pt"  # Path to your downloaded .pt file
    # INPUT_SOURCE = "/media/aryan/E89E8AAD9E8A743A1/Github/LE/yolo-world/images/"                  # Can be a single image ("test.jpg") OR a folder ("input_images/")
    INPUT_SOURCE = "/media/aryan/E89E8AAD9E8A743A1/IMAGES/"
    OUTPUT_DIR = "/media/aryan/E89E8AAD9E8A743A1/Github/LE/outputs"                      # Target directory for saved results
    CUSTOM_CLASSES = ["person", "bus", "backpack", "horse", "ball"]

    run_offline_inference(
        model_path=LOCAL_MODEL_PATH,
        source_path=INPUT_SOURCE,
        output_dir=OUTPUT_DIR,
        classes=CUSTOM_CLASSES,
        conf=0.35,
    )
