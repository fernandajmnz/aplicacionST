import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt


# Lista de imágenes en GitHub
github_images = [
    "https://github.com/fernandajmnz/aplicacionST/blob/main/Test%20Images/1.jpeg",
    "https://github.com/fernandajmnz/aplicacionST/blob/main/Test%20Images/2.jpeg",
    "https://github.com/fernandajmnz/aplicacionST/blob/main/Test%20Images/3.jpeg",
    "https://github.com/fernandajmnz/aplicacionST/blob/main/Test%20Images/4.jpeg",
    "https://github.com/fernandajmnz/aplicacionST/blob/main/Test%20Images/5.jpeg"
  
]


# Función para descargar y procesar la imagen
def load_image_from_url(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    else:
        print(f"Error al descargar la imagen: {url}")
        return None

# Función para detectar perros con filtros de imagen
def detect_dogs(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Aplicar filtro Gaussiano
    edges = cv2.Canny(blurred, 50, 150)  # Detectar bordes con Canny
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_dogs = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 5000:  # Filtrar por tamaño mínimo
            detected_dogs.append((x, y, x + w, y + h))
    
    return detected_dogs

# Procesar imágenes
def process_images(image_urls):
    for url in image_urls:
        image = load_image_from_url(url)
        if image is None:
            continue
        
        dogs = detect_dogs(image)
        
        # Dibujar detecciones
        for (x1, y1, x2, y2) in dogs:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Mostrar la imagen con detecciones
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Perros detectados: {len(dogs)}")
        plt.show()
        
        print(f"Imagen: {url}")
        print(f"Número de perros detectados: {len(dogs)}")

# Ejecutar la detección en todas las imágenes
#process_images(github_images)