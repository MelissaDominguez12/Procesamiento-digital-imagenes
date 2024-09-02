# pylint: disable=missing-module-docstring
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


def histograma(matriz):
    flattened_matrix = matriz.flatten()
    plt.hist(
        flattened_matrix,
        bins=np.arange(min(flattened_matrix), max(flattened_matrix) + 1),
        edgecolor="blue",
    )
    plt.xlabel("Número")
    plt.ylabel("Frecuencia")
    plt.title("Histograma")
    plt.show()


def img_matrix(matriz, nombre_archivo=None):
    matriz_redondeada = np.round(matriz).astype(np.uint8)
    cv2.imshow(
        "Imagen en escala de grises", matriz_redondeada
    )  # pylint: disable=no-member
    cv2.waitKey(0)  # pylint: disable=no-member
    if nombre_archivo:
        cv2.imwrite(nombre_archivo, matriz_redondeada)  # pylint: disable=no-member
    cv2.destroyAllWindows()  # pylint: disable=no-member


def expansion(matriz):
    fMax = np.max(matriz)
    fMin = np.min(matriz)
    max = 180
    min = 255

    nuevaMatriz = ((matriz - fMin) / (fMax - fMin)) * (max - min) + min
    return nuevaMatriz


def contraccion(matriz):
    fMax = np.max(matriz)
    fMin = np.min(matriz)
    cMax = 20
    cMin = 100

    nuevaMatriz = ((cMax - cMin) / (fMax - fMin)) * (matriz - fMin) + cMin
    return nuevaMatriz


def filtro_gaussiano(imagen, kernel_size=(5, 5), sigma=30):
    return cv2.GaussianBlur(imagen, kernel_size, sigma)  # pylint: disable=no-member


def agregar_ruido_gaussiano(matriz, mean=0, var=0.05):
    row, col = matriz.shape
    sigma = var**0.6
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    imagen_ruido = matriz + gauss * 255
    imagen_ruido = np.clip(imagen_ruido, 0, 255)
    return imagen_ruido.astype(np.uint8)


def cargar_imagen():
    global matriz
    filepath = filedialog.askopenfilename()
    if filepath:
        matriz = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # pylint: disable=no-member
        if matriz is not None:
            messagebox.showinfo("Cargar Imagen", "Imagen cargada exitosamente.")
        else:
            messagebox.showerror("Error", "No se pudo cargar la imagen.")
    else:
        messagebox.showwarning("Cargar Imagen", "No se seleccionó ningún archivo.")


def mostrar_histograma():
    if matriz is not None:
        histograma(matriz)
    else:
        messagebox.showerror("Error", "No se ha cargado ninguna imagen.")


def mostrar_imagen():
    if matriz is not None:
        img_matrix(matriz)
    else:
        messagebox.showerror("Error", "No se ha cargado ninguna imagen.")


def aplicar_expansion():
    global matriz_expansion
    if matriz is not None:
        matriz_expansion = expansion(matriz)
        histograma(matriz_expansion)
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(matriz, cmap="gray"), plt.title("Imagen Original")
        plt.subplot(122), plt.imshow(matriz_expansion, cmap="gray"), plt.title(
            "Imagen con Expansión del Histograma"
        )
        plt.show()
    else:
        messagebox.showerror("Error", "No se ha cargado ninguna imagen.")


def aplicar_contraccion():
    global matriz_contraccion
    if matriz is not None:
        matriz_contraccion = contraccion(matriz)
        histograma(matriz_contraccion)
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(matriz, cmap="gray"), plt.title("Imagen Original")
        plt.subplot(122), plt.imshow(matriz_contraccion, cmap="gray"), plt.title(
            "Imagen con Contracción del Histograma"
        )
        plt.show()
    else:
        messagebox.showerror("Error", "No se ha cargado ninguna imagen.")


def aplicar_ruido():
    global matriz_ruido
    if matriz is not None:
        matriz_ruido = agregar_ruido_gaussiano(matriz)
        histograma(matriz_ruido)
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(matriz, cmap="gray"), plt.title("Imagen Original")
        plt.subplot(122), plt.imshow(matriz_ruido, cmap="gray"), plt.title(
            "Imagen con Ruido Gaussiano"
        )
        plt.show()
    else:
        messagebox.showerror("Error", "No se ha cargado ninguna imagen.")


def aplicar_filtro_gaussiano():
    global matriz_gauss
    if matriz_ruido is not None:
        matriz_gauss = filtro_gaussiano(matriz_ruido)
        histograma(matriz_gauss)
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(matriz_gauss, cmap="gray"), plt.title(
            "Imagen con Filtro Gaussiano"
        )
        plt.subplot(122), plt.imshow(matriz_ruido, cmap="gray"), plt.title(
            "Imagen con Ruido Gaussiano"
        )
        plt.show()
        img_matrix(matriz_gauss, "img_gauss.jpg")
    else:
        messagebox.showerror("Error", "No se ha aplicado ruido a la imagen.")


# Configurar la interfaz gráfica de usuario
root = tk.Tk()
root.title("Procesador de Imágenes")
root.geometry("1000x1000")

# Añadir fondo
background_image = Image.open("./img/fondo.jpg")
background_image = background_image.resize((1000, 1000), resample=Image.LANCZOS)
background_photo = ImageTk.PhotoImage(background_image)


background_label = tk.Label(root, image=background_photo)
background_label.place(relwidth=1, relheight=1)

# Crear un marco para los botones con más espacio
frame = tk.Frame(root, bg="#006400", bd=5, borderwidth=10, relief="ridge")
frame.place(relx=0.5, rely=0.5, relwidth=0.75, relheight=0.75, anchor="center")

# Configurar el tamaño de los botones y la fuente
btn_config = {"width": 20, "height": 3, "bg": "#F5F5DC", "font": ("Arial", 11, "bold")}

# Título dentro del marco
titulo = tk.Label(
    frame,
    text="Segmentación de Imágenes para Detección de Incendios",
    font=("Helvetica", 16, "bold"),
    bg="#006400",
    fg="white",
)
titulo.pack(side="top", pady=10)

btn_cargar = tk.Button(frame, text="Cargar Imagen", command=cargar_imagen, **btn_config)
btn_cargar.pack(side="top", padx=5, pady=5)

btn_histograma = tk.Button(
    frame, text="Mostrar Histograma", command=mostrar_histograma, **btn_config
)
btn_histograma.pack(side="top", padx=5, pady=5)

btn_imagen = tk.Button(
    frame, text="Mostrar Imagen", command=mostrar_imagen, **btn_config
)
btn_imagen.pack(side="top", padx=5, pady=5)

btn_expansion = tk.Button(
    frame, text="Aplicar Expansión", command=aplicar_expansion, **btn_config
)
btn_expansion.pack(side="top", padx=5, pady=5)

btn_contraccion = tk.Button(
    frame, text="Aplicar Contracción", command=aplicar_contraccion, **btn_config
)
btn_contraccion.pack(side="top", padx=5, pady=5)

btn_ruido = tk.Button(
    frame, text="Aplicar Ruido Gaussiano", command=aplicar_ruido, **btn_config
)
btn_ruido.pack(side="top", padx=5, pady=5)

btn_filtro = tk.Button(
    frame,
    text="Aplicar Filtro Gaussiano",
    command=aplicar_filtro_gaussiano,
    **btn_config
)
btn_filtro.pack(side="top", padx=5, pady=5)

# Ejecutar la aplicación
root.mainloop()
