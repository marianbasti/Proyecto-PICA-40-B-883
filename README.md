# [Advancing social insect research through the development of an automated yellowjacket nest activity monitoring station using deep learning](http://img.shields.io/badge/DOI-10.1111/afe.12638-12DD11.svg)
[![DOI:10.1111/afe.12638](http://img.shields.io/badge/DOI-10.1111/afe.12638-12DD11.svg)](https://onlinelibrary.wiley.com/share/author/DH4HTNWQRZCENIATQ7P6?target=10.1111/afe.12638)
---

En este repositorio tenemos los códigos para reproducir los resultados de nuestra publicación.
Ponemos a disposición notebooks para el entrenamiento del modelo YOLOv8 de detección, inferencia y exportación de datos.

![image](https://github.com/marianbasti/Proyecto-PICA-40-B-883/assets/31198560/3e58b471-1fa7-4b2b-aa4b-a551c87173e0)

---
## Dataset
El dataset fue anotado y generado en la plataforma [Roboflow](https://roboflow.com/) anotando manualmente 1740 imágenes, aumentada a 4176 imágenes con distintas transformaciones. Separamos el dataset en un 88%-8%-4% para sets de entrenamiento-validación-test. Utilizamos las etiquetas de [avispa], [zangano] y [reina] para discriminación entre castas

El formato exportado es [Ultralytics YOLOv8](https://docs.ultralytics.com/datasets/detect/)

Enlace de descarga: [RoboFlow](https://universe.roboflow.com/ds/jdYfGOHlGu?key=JZgVkCiWdp)

---
## Adquicisión de datos
### Microcontrolador y Sistema Operativo
Utilizamos un NVIDIA Jetson Nano Dev Kit B-01 para gestionar la adquisición de datos.

### Cámara
La cámara empleada es el módulo IMX219-130.

### Sensor de Temperatura y Humedad
Para medir la temperatura y la humedad empleamos el sensor DHT-22, programado para realizar lecturas cada 5 minutos durante las filmaciones.

### Periodo de Filmación
El periodo de filmación y sensado se centra en los meses de marzo, abril y mayo, intervalos de tiempo donde se observa una mayor actividad de zánganos y reinas.

### Ubicación y Montaje
La cámara se coloca en una caja diseñada para la entrada del panal, capturando así los movimientos relevantes de la colonia. El sensor es colocado en la proximidad para capturar métricas ambientales, a una distancia suficiente del micocontrolador para evitar su interferencia de temperatura.

---
## Entrenamiento
El modelo de detección es una continuación de entrenamiento del modelo base ```yolov8.pt``` provisto por Ultralytics, con los siguientes parámetros:

```learning rate: 0.015```

```epochs: 100```

```image size: 640```

### Métricas
![results](https://github.com/marianbasti/Proyecto-PICA-40-B-883/assets/31198560/8722e928-d0f9-4ffd-b778-be478eda8701)
![confusion_matrix_normalized](https://github.com/marianbasti/Proyecto-PICA-40-B-883/assets/31198560/9941f7ff-def9-4e1f-8b6e-f7184c7bfd3a)
Por lo que podemos ver en la matriz de confusión, el modelo entrenado puede predecir correctamente del dataset [avispa] un 99%, [reina] un 96% y [zangano] un 88%.


---
## Detección
Para la detección usamos el modelo entrenado con el dataset que generamos. Exportamos los siguientes datos por cada avispa detectada:
|id   |timestamp   |temp   |humidity   |length   |width|movement|time_on_screen|video_timestamp|p_worker|p_drone|p_gyne|
|---|---|---|---|---|---|---|---|---|---|---|---|
|Identificación única   |Día y horario de la detección   |Temperatura sensada   |Humedad sensada  |Largo promedio de la avispa  |Ancho promedio de la avispa  |Si la avispa ingresó o salió. 'in' para entrada, 'out' para salida   |Cuánto tiempo tardó en entrar o salir   |Minuto y segundo de la detección en el video   |Proporcion de detección de obrera  |Proporcion de detección de  zángano  |Proporcion de detección de reina|

### Temperatura y humedad
Al tener un ritmo de sensado cada 5 minutos, estos datos asociados a cada detección son resultado de interpolación de los datos más cercanos

### Largo y ancho
Útil para determinar la casta de la avispa.
Comienza con un recuadro del área de interés generado por YOLOv8. Luego con post-procesamiento refinamos la imagen para obtener solamente el cuerpo del insecto, recortando antenas, alas y patas. Generamos el contorno de esa silueta y buscamos el rectángulo con menor área que encapsule esa silueta, para después reorientar el rectángulo y adquirir su alto y ancho. Por último, promediamos los valores a lo largo de todo el recorrido que hizo la avispa excluyendo aquellas medidas obtenidas en los extremos de la imagen (ya que el brillo es inconsistente en estas áras)

### Movimiento
En los videos podemos observar que muchas avispas merodean la entrada sin entrar ni salir. Por eso, determinamos un umbral horizontal a superar para determinar si entró (in), salió (out) o ninguna (undetermined).

### Tiempo de entrada o salida
Para el recorrido hecho, calculamos la diferencia de tiempo entre la primera y última detección.

### Clase
Durante todo el trayecto del insecto, recolectamos una predicción de clase por cada frame. Luego calculamos y guardamos que proporción de todas las detecciones corresponde a cada una.

---
## Modo de uso
### Google Colab
Ponemos a disposición un [Notebook](https://colab.research.google.com/github/marianbasti/Proyecto-PICA-40-B-883/blob/main/Notebook_PICA_40_B_883.ipynb) para facilitar tanto entrenamiento como inferencia.

### CLI
• Para entrenamiento usamos el script ```train_avispas.py``` con el siguiente comando:

```
python train_avispas.py --dataset_path "directorio del dataset en formato YOLOv8"
```

O definiendo parámetros en vez de los default:

```
python train_avispas.py --dataset_path "directorio del dataset en formato YOLOv8" --epochs 100 --lr 0.015
 ```

Encontraremos en ```runs/detect/train``` los resultados del entrenamiento.


• Para inferencia usamos el script ```detect_avispas.py``` con el siguiente comando:

```
python detect_avispas.py
```

Los siguientes argumentos están disponibles:

```
options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Modelo YOLOv8
  --input_path INPUT_PATH
                        Directorio de los videos (dd-mm-aaaa/ + sensor/)
  --save_vid            Guardar video procesado
  --csv_output CSV_OUTPUT
                        Nombre de salida para archivo CSV
  --iou IOU             IOU
  --conf CONF           confidence
  --tracker TRACKER     Tracker yaml configuration
  --distance_thresh DISTANCE_THRESH
                        Distancia mínima en píxeles entre detecciones antes de descartarlas por los conflictos que
                        genera
  --threshold THRESHOLD
                        Threshold de brillo para extraer silueta y calcular el tamaño
  --width_crop WIDTH_CROP
                        Recortar el ancho del video
  --significant_move SIGNIFICANT_MOVE
                        Que proporción del ancho del video se considera para determinar que la avispa entró o salió
  --track_discard_less_than TRACK_DISCARD_LESS_THAN
                        Descartar tracks que tengan menos de cierta cantidad de detecciones
```

---
## Observaciones y mejoras
A través de esta experimentación observamos que:

• La cantidad de cuadros por segundo tiene mucha influencia en la calidad de trackeo de movimiento. Nuestra adquisicion de datos a 10 FPS causó dificultades para seguir a los insectos que se mueven rápidamente y al determinar la entrada/salida de instectos, representa un margen de error cercano al 10%.

• Las castas deben estar igualitariamente representadas en el dataset para una mayor confianza de detección. Cuando una clase esta sobrerepresentada, tendremos un sesgo a clasificarla como tal. 

• Para detectar mejor una entrada o salida, es conveniente observar un recorrido más largo que evidencie más el movimiento del insecto. Esto se puede lograr modficando el artefacto en el que se coloca el controlador y la cámara en la entrada del nido.
