import cv2
import os
import numpy as np

# Crea una carpeta para guardar los rostros si no existe
if not os.path.exists('rostros'):
    os.makedirs('rostros')

# Carga el clasificador de rostros preentrenado de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

def obtener_modelo():
    labels = []
    faces_data = []
    label = 0

    # Cargar todos los rostros y asignar etiquetas
    for filename in os.listdir('rostros'):
        if filename.endswith('.jpg'):
            image_path = os.path.join('rostros', filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            faces_data.append(image)
            labels.append(label)
            label += 1

    if labels:
        recognizer.train(faces_data, np.array(labels))

def guardar_rostro(frame, gray, faces, contador):
    for (x, y, w, h) in faces:
        rostro = gray[y:y+h, x:x+w]
        cv2.imwrite(f'rostros/rostro_{contador}.jpg', rostro)

def detectar_y_mostrar():
    cap = cv2.VideoCapture(0)
    contador = len(os.listdir('rostros'))  # Comienza la cuenta desde los archivos existentes

    obtener_modelo()  # Entrena el modelo con los rostros existentes

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            rostro = gray[y:y+h, x:x+w]
            id, confianza = recognizer.predict(rostro)
            if confianza < 50:  # Umbral de confianza para un reconocimiento preciso
                color = (0, 255, 0)  # Verde si reconoce el rostro
            else:
                color = (0, 0, 255)  # Rojo si el rostro es desconocido

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

 # Mejora visual de la leyenda
        texto = "Presiona 'S' para guardar su rostro"
        (ancho_texto, altura_texto), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)
        x_texto = 10
        y_texto = 30
        cv2.rectangle(frame, (x_texto, y_texto - altura_texto - 10), (x_texto + ancho_texto, y_texto + 5), (50, 50, 50), -1)
        cv2.putText(frame, texto, (x_texto, y_texto), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

        cv2.imshow('Frame', frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:  # Presiona 'ESC' para salir
            break
        elif k == ord('s'):  # Presiona 's' para guardar el rostro
            guardar_rostro(frame, gray, faces, contador)
            contador += 1
            obtener_modelo()  # Re-entrena el modelo después de agregar un nuevo rostro

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detectar_y_mostrar()
