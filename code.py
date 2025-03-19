import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt



def load_image_from_url(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    else:
        print(f"Error al descargar la imagen: {url}")
        return None
