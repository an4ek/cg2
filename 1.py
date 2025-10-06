import cv2
import numpy as np
import os
import time

chars = ['#', '#', '*', '+', '=', '-', ':', '.', ' ']


def clear_screen():
    os.system('cls')


def frame_to_ascii(frame, width=120):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    height = int(gray.shape[0] * width / gray.shape[1] * 0.55)
    resized = cv2.resize(gray, (width, height))

    ascii_art = ""
    for row in resized:
        for pixel in row:
            char_index = min(int(pixel / 255 * (len(chars) - 1)), len(chars) - 1)
            ascii_art += chars[char_index]
        ascii_art += "\n"

    return ascii_art


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Не удалось открыть камеру!")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ошибка получения кадра")
                break

            ascii_frame = frame_to_ascii(frame, 120)

            clear_screen()
            print("=== ASCII ВЕБ-КАМЕРА ===")
            print("Нажмите Ctrl+C для выхода\n")
            print(ascii_frame)
            print(f"Кадр: {time.strftime('%H:%M:%S')}")

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\nВыход из программы...")
    except Exception as e:
        print(f"\nОшибка: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()