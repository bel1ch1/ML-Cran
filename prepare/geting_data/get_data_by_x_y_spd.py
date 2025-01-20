import cv2                       # Поулчение данных
import time                      # Ввявление скорости
import numpy as np               # Преобразовния
import pandas as pd              # Упаковка
from queue import Queue          # Формирование последовательностей



# Для детекции маркера
a_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
a_param = cv2.aruco.DetectorParameters()
a_param.useAruco3Detection = True
a_param.adaptiveThreshConstant = 0
marker_size = 35 / 1000 # Перевод в метры
DIST_COEF = np.array([[-0.22657406,  0.97367389, -0.00574975, -0.00995885, -1.71856769]])
MATX = np.array([[609.06873396, 0.0, 299.48555699],[0.0, 613.58229373, 238.86448876],[0.0, 0.0, 1.0]])



# Реализация специальной очереди для формирования последовательностей
class batch_queue():
    ''' Класс создает очередь и инициализирует ее нулями '''
    def __init__(self, batch_size):
        self.queue = Queue()
        for _ in range(batch_size):
            self.queue.put(None)  # Заполнение нулями определенного пространства

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


# Инициализация камеры
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
cap.set(cv2.CAP_PROP_FPS, 120)


# Инициализация очереди
que = batch_queue(80)               # указать длину одного объекта
previous_dot_X = 0                  # Создание начальной точки положения для X
previous_dot_Y = 0                  # Создание начальной точки положения для Y
first_time_dot = time.time()        # Создание начальной точки времени


while True:
    ret, frame = cap.read()           # кадр
    # Проверка работы камеры
    if not ret:
        break

    # if cv2.waitKey(1) == ord("q"):
    #     break

    else:
        # Перевод кадра в серый
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("frames", gray)   # Кратинка
        # Получение углов маркера на кадре
        corner, _, _ = cv2.aruco.detectMarkers(gray, dictionary=a_dict, parameters=a_param)
        if corner:
            # Получение вектора отклонений маркера в реальном мире
            _, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corner, markerLength=marker_size, cameraMatrix=MATX, distCoeffs=DIST_COEF
            )

            if tvec is not None:
                dot_arr = []
                # Отклонения от центра кадра по X и Y
                X = int(tvec[0][0][0]*1000)
                Y = int(tvec[0][0][1]*1000)

                dx = X - previous_dot_X                 # Получаем разницу по X
                dy = Y - previous_dot_Y                 # Получаем разницу по Y
                time_stamp = time.time()                # Временная метка
                dt = first_time_dot - time_stamp        # Получаем разницу по времени

                previous_dot_X = X                      # Обновляем точку координат
                previous_dot_Y = Y                      # Обновляем точку координат
                first_time_dot = time_stamp             # Обновляем временную точку

                distance = np.sqrt(dx**2 + dy**2)       # Вычисление дистанции
                spd = np.abs(distance / dt)             # Вычисление скорости

                dot_arr.append(X)                       # Добавление признака X
                dot_arr.append(Y)                       # Добавление признака Y
                dot_arr.append(int(spd))                # Добавление признака spd
                que.put_in_queue(dot_arr)               # Добавление массива в очередь (в объект)
                que.pull_it_out()                       # Удаление самого старого объекта очереди
                data = list(que.get_data())             # Получение полной очереди (весь объект) list
                data = [data]                           # Упаковка в массив
                df = pd.DataFrame(data=[data], columns=['coords'])

                if data[0][0] is not None:
                    print(df)
                    # Запись новых данных в CSV в режиме добавления
                    df.to_csv('datas/CSV/right_data.csv', mode='a', header=False, index=False)



# Корректное завершение программы (отключение потока камеры)
cap.release()
cv2.destroyAllWindows()
