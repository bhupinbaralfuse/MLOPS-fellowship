import pytest
from utils.preprocess import preprocess_and_infer
import os
from PIL import Image
import json

def test_preprocess_and_infer(tmp_path):
    input_dir = tmp_path / "images"
    output_file = tmp_path / "predictions.json"
    input_dir.mkdir()
    
    # Create a dummy image
    img = Image.new('RGB', (300, 300))
    img.save(input_dir / "test.jpg")
    
    # Mock model and transforms
    with open(tmp_path / "transforms.json", "w") as f:
        json.dump({
            "__version__": "0.5.1",
            "transform": {
                "__class_fullname__": "albumenations.core.composition.Compose",
                "p": 1,
                "transforms": [{"__class_fullname__": "albumenations.augmentations.transforms.Resize", "p": 1, "height": 224, "width": 224, "interpolation": 1}],
                "bbox_params": null,
                "keypoint_params": null,
                "additional_targets": {}
            }
        }, f)
    
    # Mock class mapping
    with open(tmp_path / "class_mapping.json", "w") as f:
        json.dump([{"model_idx": 1, "class_name": "test_class"}], f)
    
    # Mock MLflow setup
    os.makedirs(tmp_path / "mlruns", exist_ok=True)
    with open(tmp_path / "mlflow.db", "w") as f:
        f.write("")
    
    # Skip inference for simplicity
    preprocess_and_infer(str(input_dir), str(output_file), "http://localhost:5000")
    
    assert os.path.exists(output_file)