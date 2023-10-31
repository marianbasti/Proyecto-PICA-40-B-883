# Proyecto-PICA-40-B-883
En este repositorio tenemos los códigos para reproducir los resultados de la publicación []

Ponemos a disposición notebooks para el entrenamiento del modelo YOLOv8 de detección, inferencia y exportación de datos
![image](https://github.com/marianbasti/Proyecto-PICA-40-B-883/assets/31198560/96f06b9e-43bb-4b84-b5fa-e49684980bb0)

---
## Dataset
El dataset fue generado en la plataforma [Roboflow](https://roboflow.com/) anotando manualmente 4834 imágenes, aumentada a 9762 imágenes con distintas transformaciones. Utilizamos las etiquetas de [avispa], [zangano] y [reina] para incluir discriminación entre castas

El formato exportado es [Ultralytics YOLOv8](https://docs.ultralytics.com/datasets/detect/)

Enlace de descarga: [Google Drive](https://drive.google.com/file/d/1skVPS8g-JSSWca0zt_500vZNW5f3vPn8/view?usp=sharing)

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
Para la detección, exportamos los siguientes datos por cada avispa detectada:
|id   |timestamp   |temp   |humidity   |largo   |ancho|movement|time|filename|class|conf|
|---|---|---|---|---|---|---|---|---|---|---|
|Identificación única   |Día y horario de la detección   |Temperatura sensada   |Humedad sensada  |Largo promedio de la avispa  |Ancho promedio de la avispa  |Si la avispa ingresó o salió. 'in' para entrada, 'out' para salida   |Cuánto tiempo tardó en entrar o salir   |Nombre del video en el que se detectó a la avispa   |Clase detectada (worker, gyne, drone)  |Confianza de la detección (0-1)  |

### Temperatura y humedad
Al tener un ritmo de sensado cada 5 minutos, estos datos asociados a cada detección son resultado de interpolación de los datos más cercanos

### Largo y ancho
Útil para determinar la casta de la avispa.
Comienza con un recuadro del área de interés generado por YOLOv8. Luego con post-procesamiento refinamos la imagen para obtener solamente el cuerpo del insecto, recortando antenas, alas y patas. Generamos el contorno de esa silueta y buscamos el rectángulo con menor área que encapsule esa silueta, para después reorientar el rectángulo y adquirir su alto y ancho. Por último, promediamos los valores a lo largo de todo el recorrido que hizo la avispa excluyendo aquellas medidas obtenidas en los extremos de la imagen (ya que el brillo es inconsistente en estas áras)

### Movimiento
En los videos podemos observar que muchas avispas merodean la entrada sin entrar ni salir. Por eso, solo registramos a aquellas que superen cierto umbral de movimiento lateral para considerar que entraron o salieron del panal.

### Tiempo de entrada o salida
Para el recorrido hecho, calculamos la diferencia de tiempo entre la primera y última detección.

### Clase y confianza
Durante todo el trayecto del insecto, recolectamos todas las predicciones hechas y definimos como clase la que haya alcanzado mayor confianza.

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

O definiendo parámetros en vez de los default:

```
python train_avispas.py --model_path "directorio del modelo entrenado" --input_path "directorio con videos+data" --csv_output "nombre de salida del CSV" --save_vid --iou 0.7 --conf 0.6
```

---
## Observaciones y mejoras
A través de esta experimentación observamos que:
• La cantidad de cuadros por segundo tiene mucha influencia en la calidad de trackeo de movimiento. Nuestra adquisicion de datos a 10 FPS causó dificultades para seguir a los insectos que se mueven rápidamente y al determinar la entrada/salida de instectos, representa un margen de error cercano al 10%.
• Las castas deben estar igualitariamente representadas en el dataset para una mayor confianza de detección. Esto incluye juntar imágenes de distintos individuos. Por ejemplo, si bien incluimos a una reina en el dataset, el modelo aprendió a identificar a esa reina específica y tiene problemas para detectarlas en otros casos.
• Para detectar mejor una entrada o salida, es conveniente observar un recorrido más largo que evidencie más el movimiento del insecto. Esto se puede lograr modficando el artefacto en el que se coloca el controlador y la cámara en la entrada del nido.
