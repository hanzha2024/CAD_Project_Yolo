import torch

# model
model = torch.hub.load("./", "yolov5x", source="local")

# Images
img = "./data/images/zidane.jpg"
# batch_size = 8
# imgsz = 640

# 推理
# model = model(batch)

# inference
result = model(img)

# result show
result.show()