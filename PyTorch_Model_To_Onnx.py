import torch
import sys

sys.path.insert(0, './yolov7')

device = torch.device('cpu')
model = torch.load('best.pt', map_location=device)['model'].float()
torch.onnx.export(model, torch.zeros((1, 3, 640, 640)), 'yolov7.onnx', opset_version=12)
