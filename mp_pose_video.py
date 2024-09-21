import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Cambiar aquí el índice de la cámara (0 suele ser la cámara predeterminada)
cap = cv2.VideoCapture(2)
#Si vas usar la Camara 
up = False
down = False
count = 0

# Ajustar el tamaño de la ventana para que se vea el cuerpo completo
scale = 1.0  # Puedes ajustar este valor para redimensionar el video

with mp_pose.Pose(
    static_image_mode=False
) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks is not None:
            x1 = int(results.pose_landmarks.landmark[24].x * width)
            y1 = int(results.pose_landmarks.landmark[24].y * height)

            x2 = int(results.pose_landmarks.landmark[26].x * width)
            y2 = int(results.pose_landmarks.landmark[26].y * height)

            x3 = int(results.pose_landmarks.landmark[28].x * width)
            y3 = int(results.pose_landmarks.landmark[28].y * height)

            p1 = np.array([x1, y1])
            p2 = np.array([x2, y2])
            p3 = np.array([x3, y3])

            l1 = np.linalg.norm(p2 - p3)
            l2 = np.linalg.norm(p1 - p3)
            l3 = np.linalg.norm(p1 - p2)

            # Calcular el ángulo
            angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
            if angle >= 160:
                up = True
            if up and not down and angle <= 70:
                down = True
            if up and down and angle >= 160:
                count += 1
                up = False
                down = False

            print("contador: ", count)

            # Visualización
            aux_image = np.zeros(frame.shape, np.uint8)
            cv2.line(aux_image, (x1, y1), (x2, y2), (255, 255, 0), 8)
            cv2.line(aux_image, (x2, y2), (x3, y3), (255, 255, 0), 8)
            cv2.line(aux_image, (x1, y1), (x3, y3), (255, 255, 0), 5)
            contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
            cv2.fillPoly(aux_image, pts=[contours], color=(128, 0, 250))

            output = cv2.addWeighted(frame, 1, aux_image, 0.7, 0)

            cv2.circle(output, (x1, y1), 8, (0, 255, 255), -1)
            cv2.circle(output, (x2, y2), 8, (128, 0, 250), -1)
            cv2.circle(output, (x3, y3), 8, (255, 191, 0), -1)
            cv2.rectangle(output, (0, 0), (120, 100), (255, 255, 0), -1)
            cv2.putText(output, str(int(angle)), (x2 + 40, y2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 250), 2)
            cv2.putText(output, f"Count: {count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (128, 0, 250), 2)

            # Redimensionar el video si es necesario
            output = cv2.resize(output, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            # Mostrar el marco con el conteo y las anotaciones
            cv2.imshow("Output", output)

        # Salir del bucle si se presiona la tecla 'ESC'
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
