import cv2           # OpenCV - работа с видео и изображениями
import numpy as np   # NumPy - математические операции с массивами
import os            # Взаимодействие с операционной системой
import time          # Работа со временем и задержками

chars = ['#', '#', '*', '+', '=', '-', ':', '.', ' ']
#Символы упорядочены от самых плотных к самым светлым
#'#' - самый темный символ (представляет черные области)
#' ' - пробел, самый светлый (представляет белые области)
#Дублирование '#' дает больше веса темным оттенкам

def clear_screen(): # Очищает консоль перед выводом нового кадра
    os.system('cls')


def frame_to_ascii(frame, width=120):
    # 1. Конвертация в grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Изменение размера с сохранением пропорций
    # gray.shape[0] - исходная высота
    # gray.shape[1] - исходная ширина
    # width - желаемая ширина ASCII-арта (120 символов)
    # * 0.55 - коррекция пропорций (символы выше, чем шире)
    height = int(gray.shape[0] * width / gray.shape[1] * 0.55)
    resized = cv2.resize(gray, (width, height))

    # 3. Преобразование в ASCII
    ascii_art = ""
    for row in resized:
        for pixel in row:
            char_index = min(int(pixel / 255 * (len(chars) - 1)), len(chars) - 1)
            ascii_art += chars[char_index]
        ascii_art += "\n"

    return ascii_art


def main():
    # 1. Инициализация камеры
    # 0 - индекс камеры (обычно 0 для встроенной камеры)
    # Можно использовать 1, 2 для внешних камер
    # Возвращает объект для работы с видео потоком

    cap = cv2.VideoCapture(0)

    # 2. Проверка подключения
    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    try:
        while True:
            # Захват кадра
            ret, frame = cap.read()
            #Захват кадра: ret, frame = cap.read()
            #ret - флаг успешности (True/False)
            #frame - numpy массив с изображением
            if not ret:
                print("Ошибка получения кадра")
                break
            # Преобразование в ASCII
            ascii_frame = frame_to_ascii(frame, 120)

            # Очистка и вывод
            clear_screen()
            print(ascii_frame)
            print(f"Кадр: {time.strftime('%H:%M:%S')}")

            # Задержка
            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\nВыход из программы...")
    except Exception as e:
        print(f"\nОшибка: {e}")
    finally:
        cap.release() #закрытие подключения к камере
        cv2.destroyAllWindows() #закрытие всех окон OpenCV


if __name__ == "__main__":
    main()