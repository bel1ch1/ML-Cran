import cv2                       # Поулчение данных
import time                      # Ввявление скорости
import numpy as np               # Преобразовния
import pandas as pd              # Упаковка



# Для детекции маркера
target_marker_id = 0
a_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
a_param = cv2.aruco.DetectorParameters()
a_param.useAruco3Detection = True
a_param.adaptiveThreshConstant = 0
marker_size = 142 / 1000 # Перевод в метры
DIST_COEF = np.array([[-0.22657406,  0.97367389, -0.00574975, -0.00995885, -1.71856769]])
MATX = np.array([[609.06873396, 0.0, 299.48555699],[0.0, 613.58229373, 238.86448876],[0.0, 0.0, 1.0]])


# Инициализация камеры
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
cap.set(cv2.CAP_PROP_FPS, 120)


# Создание DataFrame для хранения данных
data = pd.DataFrame(columns=["Time", "X", "Y", "Distance", "Speed"])


# Инициализация переменных для вычислений
prev_position = None
start_time = time.time()


def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


while True:
    ret, frame = cap.read()
    # Проверка работы камеры
    if not ret:
        break


    else:
        # Перевод кадра в серый
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("frames", gray)   # Кратинка
        # Получение углов маркера на кадре
        corner, ids, _ = cv2.aruco.detectMarkers(gray, dictionary=a_dict, parameters=a_param)
        
        if ids is not None:
            for i in range(len(ids)):
                # Проверяем, является ли текущий маркер целевым
                if ids[i][0] == target_marker_id:
                    # Получение вектора отклонений маркера в реальном мире
                    _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corner, markerLength=marker_size, cameraMatrix=MATX, distCoeffs=DIST_COEF
                    )

                    # Получаем координаты маркера
                    x, y, z = tvecs[i][0]
                    current_position = (x, y, z)

                    # Вычисляем пройденное расстояние и скорость
                    if prev_position is not None:
                        distance = calculate_distance(prev_position, current_position)
                        elapsed_time = time.time() - start_time
                        speed = distance / elapsed_time if elapsed_time > 0 else 0
                    else:
                        distance = 0
                        speed = 0

                    # Сохраняем данные
                    new_row = pd.DataFrame({
                    "Time": [time.time()],
                    "X": [x],
                    "Y": [y],
                    "Z": [z],
                    "Distance": [distance],
                    "Speed": [speed]
                })
                    data = pd.concat([data, new_row], ignore_index=True)
                    # Обновляем предыдущую позицию
                    prev_position = current_position


                # Запись данных в CSV файл
        data.to_csv("T_X_Y_Z_Dist_Speed.csv", index=False)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Корректное завершение программы (отключение потока камеры)
cap.release()
cv2.destroyAllWindows()
data.to_csv("T_X_Y_Z_Dist_Speed.csv", index=False)
