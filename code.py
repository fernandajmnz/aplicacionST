import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt

# Array de Imágenes
image_links = [
    "https://i.pinimg.com/originals/93/ef/2b/93ef2beb88716fb529c8af382fe6de83.jpg",
    "https://www.thespruce.com/thmb/MdXFVicEFeOh7Oc5EG-8ormpjkQ=/750x0/filters:no_upscale():max_bytes(150000):strip_icc()/LH_KK_19036-e3c3b865bcb244fa8e6725547893588b.jpeg",
    "https://cdn.mos.cms.futurecdn.net/2ku2wNxtCWzxQXQ28LZFS7-1920-80.jpg",
    "https://th.bing.com/th/id/OIP.c4uaZmVuSiEg-gZ0UL9cTwHaFI?rs=1&pid=ImgDetMain",
    "https://cdn.britannica.com/84/206384-050-00698723/Javan-gliding-tree-frog.jpg"
]

# Cargar YOLO
weights_path = "yolov3.weights"
config_path = "yolov3.cfg"
coco_names_path = "coco.names"

net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Cargar nombres de clases
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Solo detectaremos perros y gatos
TARGET_CLASSES = ["dog", "cat"]

# Función para descargar imágenes
def load_image_from_url(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    else:
        print(f"Error al descargar la imagen: {url}")
        return None

# Función para mejorar la imagen: upscaling y filtro de nitidez
def enhance_image(image):
    # Upscale: aumenta el tamaño de la imagen (factor 2 en ancho y alto)
    upscaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Filtro de nitidez (sharpening) usando un kernel de convolución
    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(upscaled, -1, kernel_sharpen)
    return sharpened

def detect_animals(image):
    height, width = image.shape[:2]
    # Convertir la imagen para YOLO
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    classIDs = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filtrar por confianza y clases objetivo
            if confidence > 0.5 and classes[class_id] in TARGET_CLASSES:
                # Extraer bounding box
                box = detection[0:4] * np.array([width, height, width, height])
                center_x, center_y, w, h = box.astype("int")

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Almacenar la info para la fase de NMS
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(class_id)

    # Aplicar Non-Maximum Suppression para eliminar cajas duplicadas
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.3)

    final_detections = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            conf = confidences[i]
            label = classes[classIDs[i]]
            final_detections.append((label, conf, x, y, x+w, y+h))

    return final_detections

# URL de imagen de prueba
image_url = image_links[0]

# Descargar y procesar la imagen
image = load_image_from_url(image_url)
if image is not None:
    # Mejorar la imagen: upscale y aplicar filtro de nitidez
    enhanced_image = enhance_image(image)
    # Detectar animales en la imagen mejorada
    animals = detect_animals(enhanced_image)

    # Dibujar detecciones finales
    for (label, confidence, x1, y1, x2, y2) in animals:
        cv2.rectangle(enhanced_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}: {confidence:.2f}"
        cv2.putText(enhanced_image, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Mostrar imagen con detecciones
    plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Animales detectados: {len(animals)}")
    plt.show()
