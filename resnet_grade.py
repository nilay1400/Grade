import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import random
import csv
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def decompose_float32(w):
    int_repr = np.float32(w).view(np.uint32)
    sign = (int_repr >> 31) & 0x1
    exponent = (int_repr >> 23) & 0xFF
    mantissa = int_repr & 0x7FFFFF
    return sign, exponent, mantissa

def get_delta_weights(w):
    sign, exponent, mantissa = decompose_float32(w)
    delta_weights = []
    
    for bit in range(32):
        if bit == 31:  # Sign bit flip
            delta = 2.0 * abs(w)
        elif bit >= 23:  # Exponent bits
            k = bit - 23
            exponent_bits = exponent
            original_bit = (exponent_bits >> k) & 1
            
            if original_bit == 0:
                delta = abs(w) * (2.0 ** (2.0 ** k) - 1.0)
            else:
                delta = abs(w) * (1.0 - 2.0 ** (-(2.0 ** k)))
        else:  # Mantissa bits
            k = 22 - bit
            delta = 2.0 ** (exponent - 127 - (k + 1))
            
        delta_weights.append(float(delta))
    
    return delta_weights

def compute_gradients(model, test_loader):
    print("\nComputing gradients...")
    start_time = time.time()
    model.train()
    inputs, labels = next(iter(test_loader))
    inputs, labels = inputs.to(device), labels.to(device)
    model.zero_grad()
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()
    gradients = {name: param.grad.data.abs() for name, param in model.named_parameters() if 'weight' in name}
    print(f"Gradients computed in {time.time() - start_time:.2f} seconds.")
    return gradients

def save_sensitivity_scores(layer_name, weights, grads, save_dir):
    csv_file = os.path.join(save_dir, f'sensitivity_scores_{layer_name.replace(".", "_")}.csv')
    print(f"Saving sensitivity scores for layer {layer_name}...")
    start_time = time.time()

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['layer', 'index', 'bit', 'sensitivity_score'])
        writer.writeheader()

        for idx in tqdm(range(len(weights)), desc=f'Processing {layer_name}'):
            delta_weights = get_delta_weights(weights[idx])
            for bit in range(32):
                sensitivity_score = abs(grads[idx]) * abs(delta_weights[bit])
                writer.writerow({
                    'layer': layer_name,
                    'index': idx,
                    'bit': bit,
                    'sensitivity_score': sensitivity_score
                })

    print(f"Layer {layer_name} saved in {time.time() - start_time:.2f} seconds.")
    return time.time() - start_time

def calculate_layer_statistics(csv_file):
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        sensitivity_scores = [float(row['sensitivity_score']) for row in reader]

    return {
        'min': min(sensitivity_scores),
        'max': max(sensitivity_scores),
        'avg': sum(sensitivity_scores) / len(sensitivity_scores)
    }

if __name__ == "__main__":
  
    data_path = './archive/'
    model_path = 'road_0.9994904891304348.pth'
    save_dir = 'resnet_sensitivity_scores'
    os.makedirs(save_dir, exist_ok=True)

   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    
    model = torchvision.models.resnet18(pretrained=False)
    num_classes = 43  # Update based on your dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Load checkpoint - corrected version
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])  # This is the key change
    #model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    
    layer_names = [name for name, param in model.named_parameters() if 'weight' in name]
    print("\nLayers with weights:", layer_names)

    
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(os.path.join(data_path, 'Test'), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    gradients = compute_gradients(model, test_loader)

   
    layer_stats = []
    total_time = 0
    
    for layer_name in layer_names:
        print(f"\nProcessing layer: {layer_name}")
        weights = model.state_dict()[layer_name].detach().cpu().numpy().flatten()
        grad = gradients[layer_name].detach().cpu().numpy().flatten()
        
        
        min_size = min(len(weights), len(grad))
        weights = weights[:min_size]
        grad = grad[:min_size]
        
        
        layer_time = save_sensitivity_scores(
            layer_name=layer_name,
            weights=weights,
            grads=grad,
            save_dir=save_dir
        )
        total_time += layer_time
        
     
        stats = calculate_layer_statistics(
            os.path.join(save_dir, f'sensitivity_scores_{layer_name.replace(".", "_")}.csv')
        )
        layer_stats.append({
            'layer': layer_name,
            'min_sensitivity': stats['min'],
            'max_sensitivity': stats['max'],
            'avg_sensitivity': stats['avg']
        })

   
    stats_file = os.path.join(save_dir, 'layer_statistics.csv')
    with open(stats_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['layer', 'min_sensitivity', 'max_sensitivity', 'avg_sensitivity'])
        writer.writeheader()
        writer.writerows(layer_stats)
    print(f"\nLayer statistics saved to {stats_file}")
    print(f"Total processing time: {total_time:.2f} seconds")

   
    def evaluate_model(model, test_loader):
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    print("\nEvaluating original accuracy...")
    accuracy = evaluate_model(model, test_loader)
    print(f"Original Model Accuracy: {accuracy:.2f}%")