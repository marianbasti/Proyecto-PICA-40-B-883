from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description="Entrenamiento de modelo YOLOv8 para el proyecto PICA-40-B-883")
parser.add_argument('--model', default='yolov8n.pt', help='Modelo base para entrenar')
parser.add_argument('--dataset_path', default='dataset', help='Directorio del dataset en formato YOLOv8')
parser.add_argument('--epochs', default=150, help='Cantidad de epochs de entrenamiento')
parser.add_argument('--lr', default=0.015, help='Ritmo de aprendizaje')
args = parser.parse_args()

if __name__ == '__main__':
    # Definimos el directorio del dataset con formato YOLOv8
    dataset = args.dataset_path

    # Cargamos modelo base
    model = YOLO('yolov8n.pt')

    # Entrenamos el modelo con el dataset
    results = model.train(data=f'{dataset}/data.yaml', epochs=args.epochs, imgsz=640, batch=-1, lr0=args.lr)
