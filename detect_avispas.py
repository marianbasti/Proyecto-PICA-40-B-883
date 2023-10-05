import os, json, argparse
import cv2
import numpy as np
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean

from ultralytics import YOLO


# import supervision as sv
# from supervision.detection.line_counter import LineZone, LineZoneAnnotator

parser = argparse.ArgumentParser(description="YOLOv8 Tracker")

parser.add_argument('--model_path', default='best.pt', help='Modelo YOLOv8 ')
parser.add_argument('--input_path', default="E:/2023_mp4/mp4/", help='Directorio de los videos (video/ + sensordata)')
parser.add_argument('--save_vid', action='store_true', help='Guardar video procesado')
parser.add_argument('--txt_path', default='tracking', help='Nombre de salida para archivo CSV')
parser.add_argument('--iou', default=0.5, help='IOU')
parser.add_argument('--conf', default=0.5, help='confidence')

args = parser.parse_args()

# Cargamos el modelo YOLOv8
model = YOLO(args.model_path)
# Directorio de los videos (video/ + sensordata)
input_path = args.input_path
# Si guardamos video
save_vid = args.save_vid
# Output
txt_path = args.txt_path

# line_annotator = LineZoneAnnotator()

def load_sensor_data(sensor_file):
    sensor_data = {}
    with open(sensor_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            time_str = parts[1]
            temp = float(parts[2].split('=')[1][:-2])
            humidity = float(parts[3].split('=')[1][:-1])
            sensor_data[time_str] = {'temp': temp, 'humidity': humidity}
    return sensor_data

def interpolate(a, b, alpha):
    return a + alpha * (b - a)

def find_closest_data(time_str, sensor_data):
    times = sorted(sensor_data.keys())
    pos = bisect_left(times, time_str)

    if pos == 0 or pos == len(times):
        return sensor_data[times[0]] if pos == 0 else sensor_data[times[-1]]

    before_time = datetime.strptime(times[pos - 1], '%H:%M:%S')
    after_time = datetime.strptime(times[pos], '%H:%M:%S')
    current_time = datetime.strptime(time_str, '%H:%M:%S')
    
    alpha = (current_time - before_time).total_seconds() / (after_time - before_time).total_seconds()

    before = sensor_data[times[pos - 1]]
    after = sensor_data[times[pos]]

    if not (after['temp'] and after['humidity']):
        return before

    return {
        'temp': interpolate(before['temp'], after['temp'], alpha),
        'humidity': interpolate(before['humidity'], after['humidity'], alpha)
    }

def refine_single_bbox(im, bbox_xyxy):
    bbox_np = np.array(bbox_xyxy).flatten()
    x_center, y_center, w, h = int(bbox_np[0]), int(bbox_np[1]), int(bbox_np[2]), int(bbox_np[3])

    x1, y1 = x_center - (w // 2), y_center - (h // 2)
    x2, y2 = x1 + w, y1 + h
    roi = im[y1:y1+h, x1:x1+w]
    # cv2.imwrite('./avispas/roi.jpg', roi)
    # Thresholding
    try:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    except:
        print(f"Error with cropping image :\n{bbox_np}")
        
    _, thresh = cv2.threshold(roi_gray, 85, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    translated_contour = largest_contour + [x1, y1]
    cv2.polylines(annotated_frame, translated_contour, isClosed=True, color=(0, 255, 0), thickness=2)

    if len(contours) == 0:
        # box - 4 corners of the rectangle
        box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        # min_rect - center, size (width, height), angle
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        size = (x2 - x1, y2 - y1)
        angle = 0
        min_rect = (center, size, angle)
        return box, min_rect
 
    # Find the smallest enclosing rotated rectangle for closed contours
    min_rect = cv2.minAreaRect(np.vstack(largest_contour))

    # Find the smallest enclosing rotated rectangle
    min_rect = cv2.minAreaRect(np.vstack(largest_contour))
    box = cv2.boxPoints(min_rect).astype(int)
    
    # Translate to original image coordinates
    box[:, 0] += x1
    box[:, 1] += y1

    w, h = min_rect[1]

    if h > w:
        w, h = h, w
        angle = min_rect[2]
        if angle < 0:
            min_rect = (min_rect[0], (w, h), angle + 90)
        else:
            min_rect = (min_rect[0], (w, h), angle - 90)

    cv2.polylines(im, [box], isClosed=True, color=(0, 255, 0), thickness=2)

    return box, min_rect

# Inicializamos el archivo de salida
if not os.path.exists(txt_path + '.csv'):
        with open(txt_path + '.csv', 'a') as f:
            f.write('id,timestamp,temp,humidity,largo,ancho,movement,time,filename\n')

# Por cada capeta que representa un día de filmaciones
for source in os.listdir(input_path):
    
    # Extraemos la fecha
    date_str = source.split('\\')[-1]

    # Cargamos el archivo con datos de los sensores
    sensor_file = os.path.join(input_path,source, f"temperatura {date_str}.txt")
    sensor_data = load_sensor_data(sensor_file)

    # Por cada video en la carpeta "/video"
    for video_path in os.listdir(input_path+'/'+source+'/video'):

        # Inicializamos el historial de trackeos
        track_history = defaultdict(lambda: {'points': [], 'timestamp': [], 'width': [], 'height': []})

        # Abrimos el video
        cap = cv2.VideoCapture(os.path.join(input_path,source)+'/video/'+video_path)

        # Obtenemos datos del video original
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if save_vid:  # Si queremos guardar el video para visualizar el trackeo
            vid_writer = cv2.VideoWriter(f"{video_path}.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        # Limite para considerar salida/entrada
        significant_move = w/4

        # Contador de Supervision
        # LINE_START = sv.Point(w/2, 0)
        # LINE_END = sv.Point(w/2, h)
        # line_counter = sv.LineZone(start=LINE_START, end=LINE_END)

        # Extraemos la hora
        time_str = video_path.split(' ')[-1].replace('_', ':').split('.')[0]
        initial_time = datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M:%S')

        # Loop a través de cada cuadro del video
        while cap.isOpened():

            # Leemos el cuadro
            success, frame = cap.read()
            if success:
                # Corremos YOLOv8
                results = model.track(frame, persist=True, conf=args.conf, iou=args.iou, tracker='avispa_bytetrack.yaml')
                annotated_frame = results[0].plot()
                
                # Supervision
                # detections = sv.Detections.from_ultralytics(results[0])
                # line_counter.trigger(detections=detections)

                # Obtenemos cajas e IDs detectadas
                boxes = results[0].boxes.xywh.cpu()

                # Si hay detección
                if results[0].boxes:
                    if results[0].boxes.id != None:
                        # Obtenemos datos temporales de la detección
                        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        elapsed_time = timedelta(seconds=current_frame / frame_rate)
                        timestamp = (initial_time + elapsed_time).strftime('%H:%M:%S')
                        track_ids = results[0].boxes.id.int().cpu().tolist()

                        # Por cada detección
                        for box, track_id in zip(boxes, track_ids):

                            # Guardamos los puntos, timestamp y tamaños de la detección
                            x, y, w, h = box
                            track = track_history[track_id]
                            track['points'].append((float(x), float(y)))
                            track['timestamp'].append(timestamp)

                            # Refinamos el tamaño para que sea el mas compacto posible
                            refined_xyxy,min_rect = refine_single_bbox(frame,box)
                            r_w, r_h = min_rect[1]
                            track['width'].append(r_w)
                            track['height'].append(r_h)

                            # Por último dibujamos las líneas del trackeo
                            points = np.hstack(track['points']).astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 20, 0), thickness=4)

                # Mostramos la imagen
                cv2.imshow("YOLOv8 Tracking", annotated_frame)
                # cv2.imshow("YOLOv8 Tracking", line_annotator.annotate(annotated_frame, line_counter))
                if save_vid:  # Si guardamos el video procesado
                    vid_writer.write(annotated_frame)
                # Pasar al video siguiente si apretamos 'n'
                if cv2.waitKey(1) & 0xFF == ord("n"):
                    break
            else:
                # Salir del loop si terminó el video
                break
                
        # Por cada track detectado
        for id in track_history:
            # Definimos si entró o salió (entrada a la izquierda, salida a la derecha)
            initial_x = track_history[id]['points'][0][0]
            final_x = track_history[id]['points'][-1][0]
            movement = abs(final_x - initial_x)

            # Si el movimiento es significativo
            if movement > significant_move: 
                # Definimos si entró o salió
                if final_x > initial_x:
                    print(f"Object {id} moved out.")
                    str_movement='out'
                elif final_x < initial_x:
                    print(f"Object {id} moved in.")
                    str_movement='in'

                # Formateamos data
                timestamp = track_history[id]['timestamp'][0]
                data = find_closest_data(timestamp,sensor_data)
                temp = data.get('temp', 'N/A')
                humidity = data.get('humidity', 'N/A')

                # Promediamos altos y anchos
                avg_w = mean(track_history[id]['width'])
                avg_h = mean(track_history[id]['height'])

                # Calculamos cuanto tiempo tardó en entrar/salir
                time_taken = datetime.strptime(track_history[id]['timestamp'][-1],'%H:%M:%S') - datetime.strptime(track_history[id]['timestamp'][0], '%H:%M:%S')

                # Armamos el log   
                log_data=f"""{id},{timestamp},{round(temp,2)},{round(humidity,2)},{round(avg_w,2)},{round(avg_h,2)},{str_movement},{time_taken},{video_path}\n"""

                # Agregamos la data del video en el csv
                with open(f"{txt_path}.csv", "a") as f:
                    f.write(log_data)

        # Terminar si apretamos 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Finalizar     
if save_vid:
    vid_writer.release()

cap.release()
cv2.destroyAllWindows()
