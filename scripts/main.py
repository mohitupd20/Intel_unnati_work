import cv2
import numpy as np
import joblib
import os

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

optimal_threshold = 0.5

new_image_path = r'C:\Users\91706\Downloads\PixelChain-main\PixelChain-main\scripts\361BB848-95D7-4CDD-9A14-96D36FA93023.jpeg'

is_pixelated, confidence = process_single_image(new_image_path, model, optimal_threshold)

print(f'Image is {"pixelated" if is_pixelated else "non-pixelated"} with confidence {confidence:.4f}')
