import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen proporcionada
image_path = 'Imagen1.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Mostrar la imagen original
plt.imshow(image, cmap='gray')
plt.title('Imagen Original')
plt.show()

# Convertir la imagen a binaria
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
binary_image = binary_image.astype(np.uint8)

# Mostrar la imagen binaria
plt.imshow(binary_image, cmap='gray')
plt.title('Imagen Binaria')
plt.show()

def apply_erosion(image, kernel):
    erosion = cv2.erode(image, kernel, iterations=1)
    return erosion

def hit_or_miss(image, hit_kernel, miss_kernel):
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    binary_image = binary_image.astype(np.uint8)

    # Invertir la imagen para la operación "miss"
    inverted_image = cv2.bitwise_not(binary_image)
    
    # Aplicar erosión con el kernel "hit"
    erosion_hit = apply_erosion(binary_image, hit_kernel)
    # Aplicar erosión con el kernel "miss" en la imagen invertida
    erosion_miss = apply_erosion(inverted_image, miss_kernel)
    
    # Aplicar la operación de hit-or-miss (AND lógico)
    hit_or_miss_result = cv2.bitwise_and(erosion_hit, erosion_miss)
    return hit_or_miss_result

def thinning(image):
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    binary_image = binary_image.astype(np.uint8)
    skeleton = np.zeros(binary_image.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    while True:
        eroded = cv2.erode(binary_image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary_image, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary_image = eroded.copy()
        
        if cv2.countNonZero(binary_image) == 0:
            break

    return skeleton

# Definir los elementos estructurantes según la imagen proporcionada
hit_kernel = np.array([[0, 0, 0],
                       [0, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)

miss_kernel = np.array([[1, 1, 1],
                        [1, 0, 0],
                        [1, 0, 0]], dtype=np.uint8)

# Aplicar la transformada Hit-or-Miss
hit_or_miss_result = hit_or_miss(image, hit_kernel, miss_kernel)

# Guardar el resultado de Hit-or-Miss
cv2.imwrite('hit_or_miss_result.png', hit_or_miss_result)

# Aplicar adelgazamiento morfológico a la imagen original
thinning_result = thinning(image)

# Guardar el resultado del adelgazamiento morfológico
cv2.imwrite('thinning_result.png', thinning_result)

# Aplicar adelgazamiento morfológico a la imagen resultante de Hit-or-Miss
thinning_hit_or_miss_result = thinning(hit_or_miss_result)

# Guardar el resultado de aplicar ambos métodos
cv2.imwrite('thinning_hit_or_miss_result.png', thinning_hit_or_miss_result)

# Mostrar los resultados
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(hit_or_miss_result, cmap='gray')
plt.title('Resultado Hit-or-Miss')

plt.subplot(1, 3, 2)
plt.imshow(thinning_result, cmap='gray')
plt.title('Resultado Adelgazamiento')

plt.subplot(1, 3, 3)
plt.imshow(thinning_hit_or_miss_result, cmap='gray')
plt.title('Resultado Hit-or-Miss + Adelgazamiento')

plt.show()


