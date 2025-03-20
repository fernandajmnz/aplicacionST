# Semana Tec TC1001

Este proyecto es una aplicación desarrollada en Python que utiliza técnicas de visión por computadora y deep learning para detectar la presencia de perros y gatos en imágenes. El sistema está pensado para servir como base en un entorno de monitoreo remoto del hogar, permitiendo supervisar a tus mascotas de forma remota.

---

## Descripción General

El proyecto se basa en el modelo YOLO (You Only Look Once) para la detección de objetos, aprovechando su capacidad de identificar múltiples instancias de una clase en una sola pasada. Se utilizan imágenes descargadas desde URLs y se procesan mediante OpenCV. La detección se filtra para enfocarse únicamente en las clases "dog" y "cat". Además, se aplica Non-Maximum Suppression (NMS) para eliminar las detecciones duplicadas y obtener una cuenta precisa de los animales presentes en la imagen. Recientemente, se ha añadido un filtro de mejora que realiza upscaling y aplica un filtro de nitidez mediante convolución, optimizando la calidad de la imagen para un reconocimiento más preciso.

---

## Tecnologías Utilizadas

- **Python 3**  
  Lenguaje de programación principal.

- **OpenCV (cv2)**  
  Librería de visión por computadora utilizada para:
  - Leer y procesar imágenes.
  - Cargar modelos de redes neuronales (módulo dnn).
  - Mostrar resultados (dibujar rectángulos, etc.).

- **YOLO (You Only Look Once)**  
  Modelo de object detection basado en deep learning.  
  Se usan los archivos preentrenados (`yolov3.weights`, `yolov3.cfg`) para detectar objetos, incluyendo perros y gatos.

- **Coco.names**  
  Archivo que contiene las 80 clases que YOLOv3 reconoce (por ejemplo, `dog`, `cat`, `person`, etc.).

- **Requests**  
  Librería de Python para realizar solicitudes HTTP.  
  Se emplea para descargar imágenes desde una URL.

- **NumPy**  
  Librería para manipulación de arrays.  
  Se usa al decodificar la imagen descargada y para procesar las detecciones.

- **Matplotlib**  
  Librería para visualización de datos y gráficos.  
  Se utiliza para mostrar la imagen con las detecciones en Jupyter notebooks o entornos similares.

- **Non-Maximum Suppression (NMS)**  
  Proceso de filtrado que descarta detecciones repetidas de un mismo objeto, quedándose con la de mayor confianza.  
  En OpenCV se implementa con la función `cv2.dnn.NMSBoxes()`.

---

## Flujo de Trabajo (Pipeline)

### Descarga/Lectura de la Imagen
- Se obtiene la imagen desde una URL usando `requests.get`.
- Se decodifica en un array NumPy usando `cv2.imdecode`.

### Preparación de la Red YOLO
- Se cargan los pesos (`.weights`) y la configuración (`.cfg`) mediante `cv2.dnn.readNet`.
- Se leen las clases desde el archivo `coco.names`.

### Preprocesamiento y Mejora de Imagen
- **Mejora de la Imagen:**  
  La imagen se amplía mediante upscaling usando `cv2.resize` y se aplica un filtro de nitidez (sharpening) con un kernel de convolución para resaltar detalles.
- Se convierte la imagen (mejorada) a un blob usando `cv2.dnn.blobFromImage` (normalización con `1/255.0`, resize a `416x416`, etc.).
- Se pasa el blob a la red con `net.setInput(blob)`.

### Inferencia (Detección de Objetos)
- Se llama a `net.forward(output_layers)` para obtener las detecciones.
- Cada detección incluye posiciones (bounding boxes) y probabilidades para cada clase.

### Filtrado de Resultados
- Se recorre cada detección y se extrae:
  - **scores:** vector de probabilidades para cada clase.
  - **class_id:** índice de la clase con mayor puntuación.
  - **confidence:** puntuación de la detección (ej. > 0.5).
- Se verifica si la clase es “dog” o “cat” y si la confianza supera un umbral (ej. 0.5).

### Almacenamiento de Cajas y Confianzas
- Para cada detección válida, se guarda:
  - Coordenadas de la caja delimitadora (x, y, w, h).
  - Confianza (`confidence`).
  - ID de la clase (`class_id`).

### Non-Maximum Suppression (NMS)
- Se aplica `cv2.dnn.NMSBoxes()` para descartar cajas repetidas del mismo objeto que se solapan.
- Así se obtiene una lista final de detecciones limpias (una caja por objeto).

### Dibujo de Resultados
- Para cada detección aprobada por NMS, se dibuja un rectángulo en la imagen y se escribe el texto con la clase y la confianza.

### Visualización
- Se convierte la imagen de BGR a RGB (por conveniencia en Matplotlib) y se muestra con `plt.imshow()`.
- Se indica el número de animales detectados en el título de la gráfica.

### Salida / Integración
- Se puede guardar la imagen procesada, retornarla a otra función o integrarla en un sistema de vigilancia que dispare alertas si se detectan perros/gatos.

---

## Instalación

1. **Clona o descarga este repositorio.**

2. **Instala las dependencias necesarias:**

   ```bash
   pip install opencv-python numpy requests matplotlib

3. **Descarga del Modelo YOLO**

Descarga los archivos del modelo YOLO:
- `yolov3.weights`
- `yolov3.cfg`
- `coco.names`

Coloca estos archivos en la carpeta del proyecto.

## Uso

**Configura los enlaces de imágenes:**  
Edita el array `image_links` en el código para incluir las URLs de las imágenes que deseas analizar.

**Ejecuta el script:**

```bash
python nombre_del_script.py

