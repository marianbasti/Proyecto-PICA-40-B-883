import os, json, argparse, math
import cv2
import numpy as np
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean
from typing import List, Tuple

from ultralytics import YOLO

parser = argparse.ArgumentParser(description="YOLOv8 Tracker para el proyecto PICA-40-B-883")

parser.add_argument('--model_path', default='best.pt', help='Modelo YOLOv8 ')
parser.add_argument('--input_path', default="./videos/", help='Directorio de los videos (dd-mm-aaaa/ + sensor/)')
parser.add_argument('--save_vid', action='store_true', help='Guardar video procesado')
parser.add_argument('--csv_output', default='tracking', help='Nombre de salida para archivo CSV')
parser.add_argument('--iou', type=float, default=0.5, help='IOU')
parser.add_argument('--conf', type=float, default=0.5, help='confidence')
parser.add_argument('--tracker', type=str, default='avispa_bytetrack.yaml', help='Tracker yaml configuration')
parser.add_argument('--distance_thresh', type=int, default=100, help='Distancia mínima en píxeles entre detecciones antes de descartarlas por los conflictos que genera')
parser.add_argument('--threshold', type=int, default=90, help='Threshold de brillo para extraer silueta y calcular el tamaño')
parser.add_argument('--width_crop', type=int, default=1, help='Recortar el ancho del video')
parser.add_argument('--significant_move', type=int, default=0.7, help='Que proporción del ancho del video se considera para determinar que la avispa entró o salió')
parser.add_argument('--track_discard_less_than', type=int, default=None, help='Descartar tracks que tengan menos de cierta cantidad de detecciones')


args = parser.parse_args()

# Cargamos el modelo YOLOv8
model = YOLO(args.model_path)
# Directorio de los videos (video/ + sensordata)
input_path = args.input_path
# Si guardamos video
save_vid = args.save_vid
# Output
csv_output = args.csv_output
# Valor del threshold
threshold_value = args.threshold

classes_names = ['worker','gyne','drone']

from datetime import datetime
from bisect import bisect_left
import numpy as np
import cv2

def calculate_center_distance(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """
    Calcula la distancia euclidiana entre los centros de dos cuadros delimitadores.

    :param box1: Un cuadro delimitador en formato (x1, y1, x2, y2).
    :param box2: Otro cuadro delimitador en formato (x1, y1, x2, y2).
    :return: La distancia euclidiana entre los centros de los dos cuadros.
    """
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)

    distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance

def filter_close_boxes(boxes: List[Tuple[float, float, float, float]], 
                       track_ids: List[int], 
                       distance_thresh: float) -> Tuple[List[Tuple[float, float, float, float]], List[int]]:
    """
    Filtra los cuadros delimitadores eliminando ambos en cada par que están más cerca que el umbral de distancia especificado.

    :param boxes: Lista de cuadros delimitadores en formato (x1, y1, x2, y2).
    :param track_ids: Lista de identificadores de seguimiento correspondientes a cada cuadro.
    :param distance_thresh: Umbral de distancia para determinar la cercanía.
    :return: Tupla de listas filtradas de cuadros delimitadores y sus correspondientes IDs de seguimiento.
    """
    keep_flags = [True] * len(boxes)

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            dist = calculate_center_distance(boxes[i], boxes[j])
            print(f'Distancia entre deteccion: {dist}')
            if dist < distance_thresh:
                print(f'Removiendo {i} y {j}, distancia de {dist}')
                keep_flags[i] = False
                keep_flags[j] = False

    filtered_boxes = [box for k, box in enumerate(boxes) if keep_flags[k]]
    filtered_track_ids = [track_id for k, track_id in enumerate(track_ids) if keep_flags[k]]

    return filtered_boxes, filtered_track_ids
    
def load_sensor_data(sensor_file: str) -> dict:
    """
    Carga datos de un sensor desde un archivo.
    
    :param sensor_file: Nombre del archivo de sensor.
    :return: Diccionario con los datos de tiempo, temperatura y humedad.
    """
    sensor_data = {}
    try:
        with open(sensor_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                time_str = parts[1]  # String de tiempo 'HH:MM:SS.xxx'
                temp = float(parts[2].split('=')[1][:-2])  # Temperatura como float
                humidity = float(parts[3].split('=')[1][:-1])  # Humedad como float
                sensor_data[time_str] = {'temp': temp, 'humidity': humidity}
    except FileNotFoundError:
        print(f"No hay datos de sensado para el día {sensor_file}")
        sensor_data = {'temp': 'N/A', 'humidity': 'N/A'}
    return sensor_data

def interpolate(a: float, b: float, alpha: float) -> float:
    """
    Interpola linealmente entre dos valores.

    :param a: Primer valor.
    :param b: Segundo valor.
    :param alpha: Peso para la interpolación.
    :return: Valor interpolado.
    """
    return a + alpha * (b - a)

def parse_time(t: str) -> datetime:
    """
    Convierte un string de tiempo a un objeto datetime.

    :param t: String de tiempo.
    :return: Objeto datetime.
    """
    try:
        return datetime.strptime(t, '%H:%M:%S.%f')
    except ValueError:
        return datetime.strptime(t, '%H:%M:%S')

def find_closest_data(time_str: str, sensor_data: dict) -> dict:
    """
    Encuentra los datos más cercanos en tiempo a una cadena dada.

    :param time_str: Cadena de tiempo para buscar los datos más cercanos.
    :param sensor_data: Diccionario con los datos del sensor.
    :return: Diccionario con los datos de temperatura y humedad interpolados.
    """
    times = sorted(sensor_data.keys())
    pos = bisect_left(times, time_str)

    if pos == 0 or pos == len(times):
        return sensor_data[times[0]] if pos == 0 else sensor_data[times[-1]]

    before_time = parse_time(times[pos - 1])
    after_time = parse_time(times[pos])
    current_time = parse_time(time_str)

    alpha = (current_time - before_time).total_seconds() / (after_time - before_time).total_seconds()

    before = sensor_data[times[pos - 1]]
    after = sensor_data[times[pos]]

    if not (after['temp'] and after['humidity']):
        return before

    return {
        'temp': interpolate(before['temp'], after['temp'], alpha),
        'humidity': interpolate(before['humidity'], after['humidity'], alpha)
    }

def refine_single_bbox(im: np.ndarray, bbox_xyxy: list) -> tuple:
    """
    Refina una caja delimitadora en una imagen.

    :param im: Imagen como un array de numpy.
    :param bbox_xyxy: Lista con las coordenadas x e y de la caja.
    :return: Una tupla con los puntos de la caja y el rectángulo mínimo del área.
    """
    bbox_np = np.array(bbox_xyxy).flatten()
    x_center, y_center, w, h = int(bbox_np[0]), int(bbox_np[1]), int(bbox_np[2]), int(bbox_np[3])

    x1, y1 = x_center - (w // 2), y_center - (h // 2)
    x2, y2 = x1 + w, y1 + h
    roi = im[y1:y1+h, x1:x1+w]

    try:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"Error al convertir la región de interés a escala de grises: {e}")
        return None

    _, thresh = cv2.threshold(roi_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        size = (x2 - x1, y2 - y1)
        angle = 0
        min_rect = (center, size, angle)
        return box, min_rect

    largest_contour = max(contours, key=cv2.contourArea)
    translated_contour = largest_contour + [x1, y1]
    cv2.polylines(annotated_frame, translated_contour, isClosed=True, color=(0, 255, 0), thickness=3)
    min_rect = cv2.minAreaRect(np.vstack(largest_contour))
    box = cv2.boxPoints(min_rect).astype(int)

    box[:, 0] += x1
    box[:, 1] += y1

    w, h = min_rect[1]

    if h > w:
        w, h = h, w
        angle = min_rect[2]
        if angle < -45:
            min_rect = (min_rect[0], (w, h), angle + 90)
        else:
            min_rect = (min_rect[0], (w, h), angle - 90)

    return box, min_rect

# Inicializamos el archivo de salida
if not os.path.exists(csv_output + '.csv'):
        with open(csv_output + '.csv', 'a') as f:
            f.write('filename,id,date,timestamp,temp,humidity,length,width,movement,time_on_screen,video_timestamp,class,conf\n')

# Por cada capeta que representa un día de filmaciones
for source in os.listdir(input_path):
    
    # Extraemos la fecha
    date_str = source.split('\\')[-1]

    # Cargamos el archivo con datos de los sensores
    # sensor_file = os.path.join(input_path,source, f"temperatura {date_str}.txt") # formato viejo
    sensor_file = os.path.join(input_path, f"sensor/{date_str}.txt")
    sensor_data = load_sensor_data(sensor_file)

    # Por cada video en la carpeta "/video"
    # for video_path in [f for f in os.listdir(input_path+'/'+source+'/video') if f.endswith('.mp4')]: # formato viejo
    for video_path in [f for f in os.listdir(input_path+'/'+source) if f.endswith('.h264')]:

        # Inicializamos el historial de trackeos
        track_history = defaultdict(lambda: {'points': [], 'timestamp': [], 'width': [], 'height': [], 'video_timestamp': [], 'class': [], 'conf':[]})

        # Abrimos el video
        # cap = cv2.VideoCapture(os.path.join(input_path,source)+'/video/'+video_path) # formato viejo
        cap = cv2.VideoCapture(os.path.join(input_path,source)+'/'+video_path)

        # Obtenemos datos del video original
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        fps = cap.get(cv2.CAP_PROP_FPS)
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) - args.width_crop
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if save_vid:  # Si queremos guardar el video para visualizar el trackeo
            vid_writer = cv2.VideoWriter(f"{video_path.split('.mp4')[0]}.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (img_w, img_h))

        # Limite para considerar salida/entrada
        significant_move = img_w * args.significant_move

        # Limite para ignorar altos y anchos (en los bordes confunde)
        border_threshold = img_w/5

        # Extraemos la hora
        # time_str = video_path.split(' ')[-1].replace('_', ':').split('.')[0] # formato viejo
        time_str = video_path.split(' ')[1].replace('_', ':').split('.')[0]
        initial_time = datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M:%S')

        # Loop a través de cada cuadro del video
        while cap.isOpened():

            # Leemos el cuadro
            success, frame = cap.read()
            if success:

                # Recortamos el cuadro
                frame = frame[:, :-args.width_crop]

                # Corremos YOLOv8
                results = model.track(frame, persist=True, conf=args.conf, iou=args.iou, tracker=args.tracker)

                annotated_frame = results[0].plot()

                # Si hay detección
                if results[0].boxes:
                    if results[0].boxes.id != None:
                        # Obtenemos datos temporales de la detección
                        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        elapsed_time = timedelta(seconds=current_frame / frame_rate)
                        timestamp = (initial_time + elapsed_time).strftime('%H:%M:%S.%f')
                        track_ids = results[0].boxes.id.int().cpu().tolist()

                        # Obtenemos cajas e IDs detectadas
                        boxes = results[0].boxes.xywh.cpu()
                        filtered_boxes, filtered_track_ids = filter_close_boxes(boxes, track_ids, args.distance_thresh)

                        # Por cada detección
                        for idx, (box, track_id) in enumerate(zip(boxes, track_ids)):

                            # Guardamos los puntos, timestamp y tamaños de la detección
                            x, y, w, h = box
                            track = track_history[track_id]
                            track['points'].append((float(x), float(y)))
                            track['timestamp'].append(timestamp)
                            track['video_timestamp'].append(elapsed_time)

                            # Guardamos la clase de la detección
                            track['conf'].append(float(results[0].boxes.conf[idx].float().cpu().tolist()))
                            track['class'].append(results[0].boxes.cls[idx].int().cpu().tolist())

                            # Refinamos el tamaño para que sea el mas compacto posible
                            refined_xyxy,min_rect = refine_single_bbox(frame,box)
                            r_w, r_h = min_rect[1]
                            track['width'].append(r_w)
                            track['height'].append(r_h)

                            # Por último dibujamos las líneas del trackeo
                            points = np.hstack(track['points']).astype(np.int32).reshape((-1, 1, 2))
                            if x>border_threshold and x<img_w-border_threshold:
                                cv2.polylines(annotated_frame, [points], isClosed=False, color=(20, 255, 0), thickness=4)
                            else:
                                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 20, 250), thickness=4)

                # Mostramos la imagen
                cv2.imshow("YOLOv8 Tracking", annotated_frame)
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
            if args.track_discard_less_than:
                if len(track_history)<args.track_discard_less_than:
                    continue
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
            else:
                str_movement='undetermined'

            # Formateamos data
            timestamp = track_history[id]['timestamp'][0]
            data = find_closest_data(timestamp,sensor_data)
            if isinstance(data, dict):
                temp = data.get('temp', 'N/A')
                humidity = data.get('humidity', 'N/A')
            else:
                # Maneja el caso donde 'data' es una cadena de texto
                temp = 'N/A'
                humidity = 'N/A'

            # Filtramos medidas detectadas, removienlo las que estan muy cerca del borde
            filtered_widths = [w for w, (x, _) in zip(track_history[id]['width'], track_history[id]['points']) if x > border_threshold and x < (img_w - border_threshold)]
            filtered_heights = [h for h, (x, _) in zip(track_history[id]['height'], track_history[id]['points']) if x > border_threshold and x < (img_w - border_threshold)]

            # Si no hay medidas tomadas en el centro, establecemos "N/A"
            # Si hay, las promediamos
            if len(filtered_widths)>0:
                avg_w = round(mean(filtered_widths),2)
                avg_h = round(mean(filtered_heights),2)
            else:
                avg_w = "N/A"
                avg_h = "N/A"

            # Calculamos cuanto tiempo tardó en entrar/salir
            time_taken = datetime.strptime(track_history[id]['timestamp'][-1],'%H:%M:%S.%f') - datetime.strptime(track_history[id]['timestamp'][0], '%H:%M:%S.%f')

            # Clasificación vieja
            max_index = track_history[id]['conf'].index(max(track_history[id]['conf']))
            conf = round(track_history[id]['conf'][max_index],2)
            #classd = track_history[id]['class'][max_index]

            # Clasificación 
            classes = track_history[id]['class']
            classd = 0

            if classes.count(0) != len(classes):
                non_worker_classes = [cls for cls in classes if cls != 0]
                most_common_class = max(set(non_worker_classes), key=non_worker_classes.count)
                if non_worker_classes.count(most_common_class) / len(classes) >= 0.96:
                    classd = most_common_class


            # Armamos el log   
            temp_str = 'N/A' if temp == 'N/A' else f"{round(float(temp), 2)}"
            humidity_str = 'N/A' if humidity == 'N/A' else f"{round(float(humidity), 2)}"
            video_timestamp = str(track['video_timestamp'][0])[:-5]
            log_data = f"{video_path},{id},{video_path.split(' ')[0]},{str(timestamp)[:-4]},{temp_str},{humidity_str},{avg_w},{avg_h},{str_movement},{str(time_taken)[2:-4]},{video_timestamp},{classes_names[classd]},{conf}\n" #.replace(', day','')

            # Agregamos la data del video en el csv
            with open(f"{csv_output}.csv", "a") as f:
                f.write(log_data)

        # Terminar si apretamos 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Finalizar     
if save_vid:
    vid_writer.release()

cap.release()
cv2.destroyAllWindows()
