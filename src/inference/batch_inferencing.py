import os
import json
import torch
import mlflow
import logging
import torchvision
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A

logging.basicConfig(level=logging.INFO)

#################################################################
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()
latest_model = client.search_registered_models()[0].latest_versions[0]

model_path = client.download_artifacts(latest_model.run_id, "model.pt")
class_mapping_path = client.download_artifacts(latest_model.run_id, "class_mapping.json")
transforms_path = client.download_artifacts(latest_model.run_id, "transforms.json")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load(model_path).to(device)
model.eval()
####################################################################

with open(class_mapping_path) as f:
    mappings = json.load(f)
class_mapping = {item['model_idx']: item['class_name'] for item in mappings}

with open(transforms_path) as f:
    transform_config = json.load(f)
transform = A.from_dict(transform_config['transform'])

##########################################################################

class ImageDataset(Dataset):

    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)
            augmented = self.transform(image=image_np)["image"]
            tensor = torch.from_numpy(augmented).permute(2, 0, 1).float()
            return img_path, tensor
        except Exception as e:
            logging.error(f"Failed to load or transform {self.image_paths[idx]}: {e}")
            return None, None

def preprocess_and_infer(
    input_dir,
    output_file,
    batch_size=16,
    num_workers=4,
    shard_index=0,
    num_shards=1,
):


    # Shard-aware image selection
    all_images = sorted([
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    images_to_process = all_images[shard_index::num_shards]

    dataset = ImageDataset(images_to_process, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    coco_results = []
    global_image_id = shard_index * 1_000_000  # Ensure unique IDs per shard

    for batch in tqdm(dataloader, desc=f"Shard {shard_index} processing"):
        paths, tensors = batch
        if any(t is None for t in tensors):
            continue
        tensors = torch.stack(tensors).to(device)

        with torch.no_grad():
            outputs = model(tensors)

        for i in range(len(paths)):
            image_id = global_image_id + i + 1
            try:
                out = {k: v[i] for k, v in outputs.items()}
                to_keep = torchvision.ops.nms(out['pred_boxes'], out['scores'], 0.5)
                pred_boxes = out['pred_boxes'][to_keep]
                pred_classes = out['pred_classes'][to_keep]
                scores = out['scores'][to_keep]

                for bbox, label, score in zip(pred_boxes, pred_classes, scores):
                    x1, y1, x2, y2 = map(int, bbox.tolist())
                    width, height = x2 - x1, y2 - y1
                    coco_results.append({
                        'image_id': image_id,
                        'category_id': label.item(),
                        'bbox': [x1, y1, width, height],
                        'score': score.item(),
                    })
            except Exception as e:
                logging.warning(f"Error in inference for image {paths[i]}: {e}")

        global_image_id += len(paths)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(coco_results, f, indent=4)
    
    logging.info(f"Saved predictions to {output_file}")
