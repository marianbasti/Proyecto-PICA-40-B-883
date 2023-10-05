# Proyecto-PICA-40-B-883
En este repositorio tenemos los códigos para reproducir los resultados del paper []

Ponemos a disposición notebooks para el entrenamiento del modelo YOLOv8 de detección, inferencia y exportación de datos
![image](https://github.com/marianbasti/Proyecto-PICA-40-B-883/assets/31198560/96f06b9e-43bb-4b84-b5fa-e49684980bb0)

---
## Dataset
El dataset fue generado en la plataforma [Roboflow](https://roboflow.com/) anotando manualmente 4834 imágenes, aumentada a 9762 imágenes con distintas transformaciones. Utilizamos las etiquetas de [avispa], [zangano] y [reina] para incluir discriminación entre castas

El formato exportado es [Ultralytics YOLOv8](https://docs.ultralytics.com/datasets/detect/)

Enlace de descarga: [Google Drive](https://drive.google.com/file/d/1skVPS8g-JSSWca0zt_500vZNW5f3vPn8/view?usp=sharing)

---
## Adquicisión de datos
TODO: Descripción de como se adquirieron las filmaciones. Controlador, cámara, la caja, donde se colocó, en que período de tiempo, como se adquirió temperatura y humedad

---
## Entrenamiento
El modelo de detección es una continuación de entrenamiento del modelo base ```yolov8.pt``` provisto por Ultralytics, con los siguientes parámetros:

```learning rate: 0.015```

```epochs: 100```

```image size: 640```

![results](https://github.com/marianbasti/Proyecto-PICA-40-B-883/assets/31198560/8722e928-d0f9-4ffd-b778-be478eda8701)
![confusion_matrix_normalized](https://github.com/marianbasti/Proyecto-PICA-40-B-883/assets/31198560/9941f7ff-def9-4e1f-8b6e-f7184c7bfd3a)
Por lo que podemos ver en la matriz de confusión, el modelo entrenado puede predecir correctamente [avispa] un 99%, [reina] un 96% y [zangano] un 88%.


---
## Detección
Para la detección, exportamos los siguientes datos por cada avispa detectada:
|id   |timestamp   |temp   |humidity   |largo   |ancho|movement|time|filename|
|---|---|---|---|---|---|---|---|---|
|Identificación única   |Día y horario de la detección   |Temperatura sensada   |Humedad sensada  |Largo promedio de la avispa  |Ancho promedio de la avispa  |Si la avispa ingresó o salió. 'in' para entrada, 'out' para salida   |Cuánto tiempo tardó en entrar o salir   |Nombre del video en el que se detectó a la avispa   |

### Temperatura y humedad
Al tener un ritmo de sensado cada 5 minutos, estos datos asociados a cada detección son resultado de interpolación de los datos más cercanos

### Largo y ancho
Útil para determinar la casta de la avispa.
Comienza con un recuadro del área de interés generado por YOLOv8. Luego con post-procesamiento refinamos la imagen para obtener solamente el cuerpo del insecto, recortando antenas, alas y patas. Generamos el contorno de esa silueta y buscamos el rectángulo con menor área que encapsule esa silueta, para después reorientar el rectángulo y adquirir su alto y ancho. Por último, promediamos los valores a lo largo de todo el recorrido que hizo la avispa excluyendo aquellas medidas obtenidas en los extremos de la imagen (ya que el brillo es inconsistente en estas áras)

### Movimiento
En los videos podemos observar que muchas avispas merodean la entrada sin entrar ni salir. Por eso, solo registramos a aquellas que superen cierto umbral de movimiento lateral para considerar que entraron o salieron del panal.

### Tiempo de entrada o salida
Para el recorrido hecho, calculamos la diferencia de tiempo entre la primera y última detección.

