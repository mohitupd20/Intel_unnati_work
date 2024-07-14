import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import glob
import time
from tqdm import tqdm

def preprocess_image(image, downscale_factor=2):
    height, width = image.shape[:2]
    new_size = (width // downscale_factor, height // downscale_factor)
    downscaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    return downscaled_image

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

def analyze_blocks(edges, block_size=16):
    height, width = edges.shape
    block_scores = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = edges[y:y+block_size, x:x+block_size]
            if block.size == 0:
                continue
            variance = np.var(block)
            edge_density = np.sum(block) / block.size
            block_scores.append((variance, edge_density))
    return block_scores

def extract_features(image_paths, labels, downscale_factor=2, block_size=16):
    features = []
    target = []
    for image_path, label in tqdm(zip(image_paths, labels), total=len(image_paths), desc="Extracting Features"):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Image at path {image_path} could not be loaded.")
            continue
        downscaled_img = preprocess_image(img, downscale_factor)
        edges = detect_edges(downscaled_img)
        block_scores = analyze_blocks(edges, block_size)
        features.extend(block_scores)
        target.extend([label] * len(block_scores))
    return np.array(features), np.array(target)

pixelated_image_paths = glob.glob('C:/Users/91706/Downloads/PixelChain-main/PixelChain-main/scripts/depixelated images/*')
non_pixelated_image_paths = glob.glob('C:/Users/91706/Downloads/PixelChain-main/PixelChain-main/scripts/pixelated image/*')

image_paths = pixelated_image_paths + non_pixelated_image_paths
labels = [1] * len(pixelated_image_paths) + [0] * len(non_pixelated_image_paths)

X, y = extract_features(image_paths, labels)

if X.size == 0 or y.size == 0:
    raise ValueError("No valid images found. Please check the image paths and ensure the images exist.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

model = LogisticRegression(max_iter=1000, class_weight=class_weight_dict)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

optimal_idx = np.argmax(2 * (precisions * recalls) / (precisions + recalls))
optimal_threshold = thresholds[optimal_idx]

y_pred = (y_pred_proba >= optimal_threshold).astype(int)

print(f'F1 Score: {f1_score(y_test, y_pred)}')

os.makedirs('model', exist_ok=True)
model_path = os.path.join('model', 'pixelation_model.pkl')
joblib.dump(model, model_path)
print(f'Model saved to {model_path}')

def process_single_image(image_path, model, optimal_threshold):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    downscaled_img = preprocess_image(img)
    edges = detect_edges(downscaled_img)
    block_scores = analyze_blocks(edges)
    
    if not block_scores:
        raise ValueError(f"No blocks found in image {image_path}")
    
    block_scores = np.array(block_scores)
    predictions = model.predict_proba(block_scores)[:, 1]
    
    average_prediction = np.mean(predictions)
    
    is_pixelated = average_prediction >= optimal_threshold
    return is_pixelated, average_prediction

model_path = os.path.join('model', 'pixelation_model.pkl')
model = joblib.load(model_path)

new_image_path = 'C:/Users/91706/Downloads/PixelChain-main/PixelChain-main/scripts/luxa.org-pixelate-3D30DB79-DCC6-438A-A85F-271BAE58382C (5).jpeg'

is_pixelated, confidence = process_single_image(new_image_path, model, optimal_threshold)

print(f'Image is {"pixelated" if is_pixelated else "non-pixelated"} with confidence {confidence:.4f}')
