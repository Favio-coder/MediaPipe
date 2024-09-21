import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import math

# Inicializar mediapipe pose como clase
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Configurar la función de pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    model_complexity=1
)

# Función para detectar pose en una imagen
def detectarPose(image, pose):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []

    if results.pose_landmarks:
        # Dibujar landmarks en la imagen con colores fluorescentes
        mp_drawing.draw_landmarks(
            image=output_image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=5, circle_radius=5),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
        )
        
        # Iterar sobre todos los landmarks
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), int(landmark.z * width)))
    
    return output_image, landmarks

# Función para calcular el ángulo entre tres landmarks
def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360

    return angle

# Clasificar la pose basada en los ángulos
def classifyPose(landmarks):
    label = "Pose no detectada"
    color = (0, 0, 255)  # Rojo para "Pose no detectada"

    # Calcula los ángulos requeridos
    if len(landmarks) >= 33:
        left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

        right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

        left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

        right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

        left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

        right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

        # Clasificación de poses
        if (165 < left_elbow_angle < 195 and 165 < right_elbow_angle < 195) and \
           (80 < left_shoulder_angle < 110 and 80 < right_shoulder_angle < 110):
            if (165 < left_knee_angle < 250 and 90 < left_knee_angle < 250):
                label = "Warrior II Pose"
                color = (0, 255, 0)  # Verde para "Pose detectada"
            elif (165 < right_knee_angle < 250 and 90 < right_knee_angle < 250):
                label = "Warrior II Pose"
                color = (0, 255, 0)  # Verde para "Pose detectada"
            elif 160 < left_knee_angle < 195 and 160 < right_knee_angle < 195:
                label = "T Pose"
                color = (0, 255, 0)  # Verde para "Pose detectada"

        elif (165 < left_knee_angle < 195 or 165 < right_knee_angle < 195) and \
             (315 < left_knee_angle < 335 or 25 < right_knee_angle < 45):
            label = "Tree Pose"
            color = (0, 255, 0)  # Verde para "Pose detectada"

    return label, color

# Convertir RGB a hexadecimal
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

# Función para actualizar la interfaz gráfica
def update_frame():
    ok, frame = camera_video.read()
    if not ok:
        return

    frame = cv2.flip(frame, 1)
    frame, landmarks = detectarPose(frame, pose)

    if landmarks:
        label, color = classifyPose(landmarks)
    else:
        label = "Pose no detectada"
        color = (0, 0, 255)  # Rojo para "Pose no detectada"

    # Convertir el color RGB a hexadecimal
    hex_color = rgb_to_hex(color)

    # Limpiar el canvas y dibujar el fondo y el texto de estado
    canvas.delete("all")
    canvas.create_rectangle(0, 0, 1280, 70, fill="black")
    canvas.create_text(640, 35, text=label, fill=hex_color, font=("Helvetica", 24, "bold"))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_tk = ImageTk.PhotoImage(image=frame_pil)

    panel_image.config(image=frame_tk)
    panel_image.image = frame_tk

    root.after(10, update_frame)

# Inicializar la ventana de tkinter
root = tk.Tk()
root.title("Clasificador de Poses")

# Crear el panel para mostrar la imagen de la cámara
panel_image = tk.Label(root)
panel_image.pack()

# Crear un lienzo para el texto del estado de la pose
canvas = tk.Canvas(root, width=1280, height=70, bg="white")
canvas.pack()

# Inicializar la cámara
camera_video = cv2.VideoCapture(0)
# El valor 0 es para usar la camara de la computadora
# El valor 1 es para usar camaras diferentes
# Usen siempre el valor 2, si van usar las camaras de su celular
camera_video.set(3, 1280)
camera_video.set(4, 960)

update_frame()

# Ejecutar el loop principal de tkinter
root.mainloop()

camera_video.release()
cv2.destroyAllWindows()
