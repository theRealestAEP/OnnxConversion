from ultralytics import YOLO
import onnxruntime as ort
import numpy as np
import torch


# Load the models
model = YOLO("yolo11n.pt")

model.export(format="onnx", opset=12, simplify=True, dynamic=True, half=True)


onnx_model = YOLO("yolo11n.onnx")

# Run inference
image_path = "https://ultralytics.com/images/bus.jpg"
original_results = model(image_path)
results = onnx_model(image_path)

# Compare results
print("Original model boxes:")
print(original_results[0].boxes.xyxy)
print("\nONNX model boxes:")
print(results[0].boxes.xyxy)

# Calculate differences
diff = torch.abs(original_results[0].boxes.xyxy - results[0].boxes.xyxy)
print("\nAbsolute differences:")
print(diff)

print(f"\nMax difference: {torch.max(diff).item()}")
print(f"Mean difference: {torch.mean(diff).item()}")

try:
    assert np.allclose(results[0].boxes.xyxy.cpu().numpy(), 
                       original_results[0].boxes.xyxy.cpu().numpy(), 
                       atol=1e-4)
    print("ONNX export successful and validated!")
except AssertionError:
    print("Assertion failed. Results do not match closely enough.")