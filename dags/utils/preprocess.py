import os
import json
import torch
import mlflow


import torchvision
import numpy as np


from PIL import Image
import albumentations as A

def preprocess_and_infer(input_dir, output_file, mlflow_tracking_uri):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Fetch latest model from MLflow
    client = mlflow.tracking.MlflowClient()
    latest_model = client.search_registered_models()[0].latest_versions[0]
    model_path = client.download_artifacts(latest_model.run_id, "model.pt")
    class_mapping_path = client.download_artifacts(latest_model.run_id, "class_mapping.json")
    transforms_path = client.download_artifacts(latest_model.run_id, "transforms.json")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(model_path).to(device)
    
    with open(class_mapping_path) as f:
        mappings = json.load(f)
    class_mapping = {item['model_idx']: item['class_name'] for item in mappings}
    
    with open(transforms_path) as f:
        transform_config = json.load(f)
    transform = A.from_dict(transform_config['transform'])
    
    coco_results = []
    image_id = 1
    
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        image = Image.open(img_path)
        image = np.array(image)
        augmented = transform(image=image)['image']
        
        x = torch.from_numpy(augmented).to(device).permute(2, 0, 1).float()
        with torch.no_grad():
            y = model(x)
            to_keep = torchvision.ops.nms(y['pred_boxes'], y['scores'], 0.5)
            pred_boxes = y['pred_boxes'][to_keep]
            pred_classes = y['pred_classes'][to_keep]
            scores = y['scores'][to_keep]
            
            for bbox, label, score in zip(pred_boxes, pred_classes, scores):
                bbox = list(map(int, bbox))
                x1, y1, x2, y2 = bbox
                width, height = x2 - x1, y2 - y1
                coco_results.append({
                    'image_id': image_id,
                    'category_id': label.item(),
                    'bbox': [x1, y1, width, height],
                    'score': score.item(),
                })
        image_id += 1
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(coco_results, f, indent=4)