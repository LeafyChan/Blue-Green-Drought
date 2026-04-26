import onnxruntime as ort
import numpy as np

# 1. Load the model session
# For Jetson, you can specify ['TensorrtExecutionProvider', 'CUDAExecutionProvider']
session = ort.InferenceSession("checkpoints/gwsat_int8.onnx")

# 2. Prepare the input data
# GWSat expects a float32 tensor of shape (Batch, Bands, Height, Width)
# The band order MUST be: [B4, B5, B6, B7, B8, B8A, B11, B12]
dummy_input = np.random.randn(1, 8, 64, 64).astype(np.float32)

# 3. Run inference
# 's2_patch' is the input name defined in export.py
outputs = session.run(None, {"s2_patch": dummy_input})

# 4. Interpret the result
logits = outputs[0]
predicted_class = np.argmax(logits)
labels = ["Stable", "Moderate", "Critical"]

print(f"Prediction: {labels[predicted_class]} (Raw logits: {logits})")