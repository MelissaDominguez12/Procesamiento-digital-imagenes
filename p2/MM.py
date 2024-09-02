import cv2
import numpy as np

def apply_erosion(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error al leer la imagen: {image_path}")

    kernel = np.ones((7, 7), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=1)
    return erosion

def apply_dilation(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error al leer la imagen: {image_path}")

    kernel = np.ones((7, 7), np.uint8)
    dilation = cv2.dilate(image, kernel, iterations=1)
    return dilation

def apply_closing(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error al leer la imagen: {image_path}")

    kernel = np.ones((8, 8), np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closing

def apply_opening(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error al leer la imagen: {image_path}")

    kernel = np.ones((10, 10), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opening

def apply_gradient(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error al leer la imagen: {image_path}")

    kernel = np.ones((5, 5), np.uint8)
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return gradient

def apply_top_hat(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error al leer la imagen: {image_path}")

    kernel = np.ones((9, 9), np.uint8)
    top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    return top_hat

def apply_bottom_hat(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error al leer la imagen: {image_path}")

    kernel = np.ones((9, 9), np.uint8)
    bottom_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    return bottom_hat



# Lista de rutas de las imágenes y operaciones correspondientes
image_operations = [
    ('img/gris-1.jpg', apply_erosion, 'erosion'),
    ('img/gris-2.jpg', apply_dilation, 'dilation'),
    ('img/gris-3.jpg', apply_closing, 'closing'),
    ('img/gris-4.jpg', apply_opening, 'opening'),
    ('img/gris-1.jpg', apply_gradient, 'gradient'),
    ('img/gris-2.jpg', apply_top_hat, 'top_hat'),
    ('img/gris-3.jpg', apply_bottom_hat, 'bottom_hat')
]

# Aplicar las operaciones a cada imagen
for i, (path, operation, op_name) in enumerate(image_operations):
    try:
        result = operation(path)
        
        # Guardar el resultado
        cv2.imwrite(f'{op_name}_{i+1}.jpg', result)
        
        # Imprimir mensaje de éxito
        print(f'{op_name.capitalize()} applied to {path} and saved as {op_name}_{i+1}.jpg')

    except ValueError as e:
        print(e)