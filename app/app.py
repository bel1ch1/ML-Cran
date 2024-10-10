import numpy as np
import cv2
import time
from queue import Queue
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --- Константы ---
a_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
a_param = cv2.aruco.DetectorParameters()
marker_size = 35 / 1000  # Перевод в метры

DIST_COEF = np.array([[-0.22657406,  0.97367389, -0.00574975, -0.00995885, -1.71856769]])
MATX = np.array([[609.06873396, 0.0, 299.48555699],[0.0, 613.58229373, 238.86448876],[0.0, 0.0, 1.0]])

# Загрузка модели
model = load_model('rnn_lstm_model_2.h5')

# --- Реализация специальной очереди для данных ---
class batch_queue():
    ''' Класс создает очередь и инициализирует её None значениями '''
    def __init__(self, batch_size):
        self.queue = Queue(maxsize=batch_size)
        for _ in range(batch_size):
            self.queue.put(None)  # Заполнение очереди None значениями

    # Добавление нового элемента
    def put_in_queue(self, coords):
        if self.queue.full():
            self.queue.get()  # Удаление старого элемента
        self.queue.put(coords)  # Добавление нового элемента

    # Возвращает пакет в виде numpy массива
    def get_data(self):
        data = list(self.queue.queue)  # Получаем все элементы из очереди
        data = [d for d in data if d is not None]  # Убираем None элементы
        return np.array(data)

# Инициализация камеры
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
cap.set(cv2.CAP_PROP_FPS, 120)

# Инициализация очереди
que = batch_queue(80)  # Длина одного пакета последовательностей
previous_dot_X = 0  # Создание начальной точки положения для X
previous_dot_Y = 0  # Создание начальной точки положения для Y
first_time_dot = time.time()  # Создание начальной точки времени

# Нормализация входных данных (как это было сделано при обучении)
scaler = MinMaxScaler()

while True:
    ret, frame = cap.read()  # Получение кадра
    if not ret:
        break

    # Перевод кадра в серый
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Получение углов маркера на кадре
    corners, _, _ = cv2.aruco.detectMarkers(gray, dictionary=a_dict, parameters=a_param)

    if corners:
        # Получение вектора отклонений маркера в реальном мире
        _, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, markerLength=marker_size, cameraMatrix=MATX, distCoeffs=DIST_COEF
        )

        if tvec is not None:
            dot_arr = []
            # Отклонения от центра кадра по X и Y
            X = int(tvec[0][0][0] * 1000)
            Y = int(tvec[0][0][1] * 1000)

            # Вычисление изменений по X и Y
            dx = X - previous_dot_X
            dy = Y - previous_dot_Y

            # Временная метка
            time_stamp = time.time()
            dt = time_stamp - first_time_dot  # Получаем разницу по времени

            # Обновление предыдущих точек и времени
            previous_dot_X = X
            previous_dot_Y = Y
            first_time_dot = time_stamp

            # Вычисление расстояния и скорости
            distance = np.sqrt(dx ** 2 + dy ** 2)
            spd = np.abs(distance / dt)

            # Добавление признаков X, Y и скорости в массив
            dot_arr.append(X)
            dot_arr.append(Y)
            dot_arr.append(spd)

            # Добавление данных в очередь
            que.put_in_queue(dot_arr)

            # Получение полного пакета данных из очереди
            data = que.get_data()

            # Преобразование данных для модели
            if len(data) == 80:  # Проверка, что очередь заполнена
                # Нормализация данных перед подачей в модель
                data = scaler.fit_transform(data)
                data = data.reshape(1, data.shape[0], data.shape[1])  # (1, 80, 3)

                # Предсказание модели
                predictions = model.predict(data,verbose=0)
                p = predictions
                if predictions[0][0] > 0.5:
                    print("DA")
                else:
                    print("NET")
