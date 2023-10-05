from ultralytics import YOLO

# Definimos el directorio del dataset con formato YOLOv8
dataset = 'dataset'

# Cargamos modelo base
model = YOLO('yolov8n.pt')

# Entrenamos el modelo con el dataset
results = model.train(data=f'{dataset}/data.yaml', epochs=150, imgsz=640, batch=-1, lr0=0.015)