from queue import Queue

import cv2
import numpy as np



a_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
a_param = cv2.aruco.DetectorParameters()
marker_size = 35 / 1000 # Перевод в метры
DIST_COEF = np.array([[-0.22657406,  0.97367389, -0.00574975, -0.00995885, -1.71856769]])
MATX = np.array([[609.06873396, 0.0, 299.48555699],[0.0, 613.58229373, 238.86448876],[0.0, 0.0, 1.0]])



class batch_queue():
    '''Класс создает очередь и инициализирует ее нулями'''
    def __init__(self, batch_size):
        self.queue = Queue()
        for _ in range(batch_size):
            self.queue.put(0)

    # Добавление нового элемента
    def put_in_queue(self, coords):
        self.queue.put(coords)

    # Удаление старого элемента
    def pull_it_out(self):
        self.queue.get()

    # Возвращает пакет
    def get_data(self):
        self.data = self.queue.queue
        return self.data

    def speedometer(self):
        '''Входные данные должны иметь вид:
        [
        [time1, time2, time3 ...],
        [X1, X2, X3],
        [Y1, Y2, Y3]
        ]
        Данная функция принимает массив данных и возвращает направление в градусах(direction),
        и скорость(speed) в м/сек.'''
        # Парсинг данных
        T = self.data[0] # Время
        X = self.data[1] # X - координаты
        Y = self.data[2] # Y - координаты

        # Вычисляем разности последовательностей
        dt = np.diff(T)
        dx = np.diff(X)
        dy = np.diff(Y)

        # Вычисляем направление
        angle = np.arctan2(dy, dx)
        direction = np.degrees(angle)

        # Вычисляем скорость
        vx = dx / dt
        vy = dy / dt
        speed = np.sqrt(vx**2 + vy**2)

        return [[speed],[direction]]


cap = cv2.VideoCapture(0)
que = batch_queue()

while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corner, _, _ = cv2.aruco.detectMarkers(gray, dictionary=a_dict, parameters=a_param)

        if corner:
            _, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corner, markerLength=marker_size, cameraMatrix=MATX, distCoeffs=DIST_COEF
            )

            if tvec:
                # Дистанции
                dX = int(tvec[0][0][0])
                dY = int(tvec[0][0][1])
