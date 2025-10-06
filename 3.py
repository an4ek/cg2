import cv2 # OpenCV - основная библиотека для работы с изображениями
import numpy as np # NumPy - математические операции и работа с массивами
from sklearn.cluster import KMeans # Алгоритм кластеризации K-means
import matplotlib.pyplot as plt # Создание графиков и визуализаций
import matplotlib.patches as patches # Геометрические фигуры для легенды
from PIL import Image, ImageDraw, ImageFont # Растровая графика и работа со шрифтами
import os # Работа с файловой системой
import argparse # Парсинг аргументов командной строки

# Палитра DMC - стандартные цвета ниток для вышивки
DMC_COLORS = {
    'White': (255, 255, 255),
    'Ecru': (240, 234, 218),
    'Black': (0, 0, 0),
    'DK Gray': (85, 85, 85),
    'MD Gray': (128, 128, 128),
    'LT Gray': (200, 200, 200),

    # Красные
    'Red': (255, 0, 0),
    'DK Red': (190, 0, 0),
    'Bright Red': (255, 50, 50),
    'Crimson': (220, 20, 60),

    # Розовые
    'Pink': (255, 192, 203),
    'DK Pink': (231, 84, 128),
    'LT Pink': (255, 182, 193),

    # Оранжевые
    'Orange': (255, 165, 0),
    'DK Orange': (255, 140, 0),
    'Coral': (255, 127, 80),

    # Желтые
    'Yellow': (255, 255, 0),
    'DK Yellow': (255, 215, 0),
    'LT Yellow': (255, 255, 200),

    # Зеленые
    'Green': (0, 255, 0),
    'DK Green': (0, 100, 0),
    'Bright Green': (0, 255, 100),
    'Forest Green': (34, 139, 34),
    'LT Green': (144, 238, 144),

    # Синие
    'Blue': (0, 0, 255),
    'DK Blue': (0, 0, 139),
    'Bright Blue': (0, 100, 255),
    'Royal Blue': (65, 105, 225),
    'LT Blue': (173, 216, 230),

    # Фиолетовые
    'Purple': (128, 0, 128),
    'DK Purple': (75, 0, 130),
    'Violet': (238, 130, 238),
    'Lavender': (230, 230, 250),

    # Коричневые
    'Brown': (165, 42, 42),
    'DK Brown': (101, 67, 33),
    'LT Brown': (210, 180, 140),
    'Tan': (210, 180, 140),
}


class CrossStitchConverter:
    def __init__(self, dmc_colors=DMC_COLORS):
        """Инициализация конвертера с палитрой цветов DMC"""
        # Сохраняем палитру цветов DMC
        self.dmc_colors = dmc_colors
        # Преобразуем цвета в numpy массив для быстрых вычислений
        self.dmc_colors_rgb = np.array(list(dmc_colors.values()))
        # Сохраняем названия цветов
        self.dmc_names = list(dmc_colors.keys())
        #dmc_colors: Словарь с названиями и RGB значениями ниток
        #dmc_colors_rgb: Преобразование в NumPy массив для векторных операций
        #dmc_names: Отдельное хранение названий для быстрого доступа

    def load_image(self, image_path):
        """Загрузка изображения"""
        # Проверка существования файла
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл {image_path} не найден")

        # Загрузка изображения с помощью OpenCV
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Не удалось загрузить изображение {image_path}")

        # Конвертация из BGR (формат OpenCV) в RGB
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        self.image = self.enhance_image(self.image)

        print(f" Изображение загружено: {self.image.shape[1]}×{self.image.shape[0]} пикселей")
        return self.image.shape[:2]

    def enhance_image(self, image):
        """Улучшение качества изображения для лучшей детализации"""
        # Фильтр увеличения резкости
        # Сверточная матрица 3x3: Лапласиан для выделения границ
        # Принцип работы: Усиление высокочастотных компонентов изображения
        # Результат: Более четкие контуры и детали
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)

        # Работа в LAB цветовом пространстве
        # L (Lightness): Яркостная компонента
        # A: Цветовая ось зеленый-красный
        # B: Цветовая ось синий-желтый
        # Разделение яркости и цвета позволяет улучшать контраст без искажения цветов
        lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Адаптивное выравнивание гистограммы
        # Адаптивность: Обработка отдельных регионов изображения
        # Ограничение контраста: Предотвращение усиления шума
        # Параметры:
        # clipLimit=2.0: Максимальное усиление контраста
        # tileGridSize=(8,8): Размер регионов для обработки
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Обратное преобразование
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return enhanced

    def resize_image(self, max_stitches):
        h, w = self.image.shape[:2]
        #Определение ограничивающей размерности (ширина или высота)
        #Масштабирование второй размерности с сохранением соотношения сторон
        #Гарантия, что ни одна размерность не превысит max_stitches
        # Сохранение пропорций
        if w > h:
            new_w = min(max_stitches, w)
            new_h = int(h * new_w / w)
        else:
            new_h = min(max_stitches, h)
            new_w = int(w * new_h / h)

        new_w = max(new_w, 10)
        new_h = max(new_h, 10)

        print(f"📐 Изменение размера: {w}×{h} → {new_w}×{new_h} крестиков")

        # INTER_CUBIC интерполяция(Бикубическая интерполяция, высокое качество для уменьшения)
        self.image_resized = cv2.resize(self.image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return new_w, new_h

#Уменьшение цветов
    def reduce_colors(self, max_colors):
        # Подготовка данных для кластеризации
        # reshape(-1, 3): Преобразование 2D изображения в 1D массив пикселей
        # Каждый пиксель представляется как точка в 3D RGB пространстве
        pixels = self.image_resized.reshape(-1, 3)

        n_colors = min(max_colors, len(self.dmc_colors))
        print(f"Кластеризация на {n_colors} цветов...")

        # K-means кластеризация
        # n_clusters: Количество желаемых цветов в палитре
        # random_state=42: Фиксированный seed для воспроизводимости результатов
        # n_init=5: Количество случайных инициализаций (выбор лучшей)
        # max_iter=300: Максимальное количество итераций для сходимости
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=5, max_iter=300)
        labels = kmeans.fit_predict(pixels)

        reduced_colors = []
        color_mapping = {}

        # Замена на цвета DMC
        for i, center in enumerate(kmeans.cluster_centers_):
            distances = np.linalg.norm(self.dmc_colors_rgb - center, axis=1)
            closest_idx = np.argmin(distances)
            dmc_color = self.dmc_colors_rgb[closest_idx]
            dmc_name = self.dmc_names[closest_idx]

            reduced_colors.append(dmc_color)
            color_mapping[i] = (dmc_color, dmc_name)

        self.reduced_image = np.array([reduced_colors[label] for label in labels])
        self.reduced_image = self.reduced_image.reshape(self.image_resized.shape)

        self.reduced_image = cv2.medianBlur(self.reduced_image.astype(np.uint8), 1)

        self.color_labels = labels.reshape(self.image_resized.shape[:2])
        self.color_mapping = color_mapping
        self.actual_colors = n_colors

        return reduced_colors

#Генерация символов
    def get_symbols(self, num_symbols):
        symbols = [
            '■', '□', '▲', '△', '●', '○', '★', '☆', '♦', '♢',
            '♥', '♡', '♣', '♤', '♠', '♧', '✓', '✗', '✶', '✷',
            '✸', '✹', '✺', '✻', '✼', '✽', '✾', '✿', '❀', '❁'
        ]

        # Дополнение буквами при необходимости
        if num_symbols > len(symbols):
            symbols.extend([chr(i) for i in range(65, 91)])  # A-Z Заглавные буквы
        if num_symbols > len(symbols):
            symbols.extend([chr(i) for i in range(97, 123)])  # a-z Строчные буквы

        return symbols[:num_symbols]

    #Создание схемы
    def create_stitch_chart(self, output_path, grid_spacing=10):
        h, w = self.color_labels.shape
        # Расчет размеров
        # cell_size=25: Размер ячейки для одного крестика в пикселях
        cell_size = 25

        img_width = w * cell_size + (w + 1) #Учет границ между ячейками
        img_height = h * cell_size + (h + 1)
        # Создание изображения
        pattern_image = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(pattern_image)

        try:
            font = ImageFont.truetype("arial.ttf", cell_size - 10)
        except:
            font = ImageFont.load_default()

        unique_colors = np.unique(self.color_labels)
        symbols = self.get_symbols(len(unique_colors))
        symbol_map = {}

        for i, color_idx in enumerate(unique_colors):
            symbol_map[color_idx] = symbols[i]

        for y in range(h):
            for x in range(w):
                color_idx = self.color_labels[y, x]
                dmc_color, dmc_name = self.color_mapping[color_idx]
                symbol = symbol_map[color_idx]

                x_pos = x * cell_size + x + 1
                y_pos = y * cell_size + y + 1
                # Цветной фон
                draw.rectangle([x_pos, y_pos, x_pos + cell_size, y_pos + cell_size],
                               fill=tuple(dmc_color), outline='lightgray')

                bbox = draw.textbbox((0, 0), symbol, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                text_x = x_pos + (cell_size - text_width) // 2
                text_y = y_pos + (cell_size - text_height) // 2
                # Контрастный текст
                brightness = np.mean(dmc_color)
                text_color = 'black' if brightness > 128 else 'white'

                draw.text((text_x, text_y), symbol, fill=text_color, font=font)

        for x in range(0, w, grid_spacing):
            line_x = x * (cell_size + 1)
            if line_x < img_width:
                draw.line([(line_x, 0), (line_x, img_height)], fill='gray', width=1)
                draw.text((line_x + 5, 5), str(x + 1), fill='darkgray', font=font)

        for y in range(0, h, grid_spacing):
            line_y = y * (cell_size + 1)
            if line_y < img_height:
                draw.line([(0, line_y), (img_width, line_y)], fill='gray', width=1)
                draw.text((5, line_y + 5), str(y + 1), fill='darkgray', font=font)

        pattern_image.save(output_path, dpi=(300, 300))
        print(f"Схема сохранена: {output_path}")
        return symbol_map

#Создание легенды
    def create_legend(self, symbol_map, output_path):
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        ax.text(0.1, 0.95, 'ЛЕГЕНДА СХЕМЫ ВЫШИВКИ', fontsize=16,
                fontweight='bold', ha='left', va='top')

        ax.text(0.1, 0.92, f'Количество цветов: {self.actual_colors}',
                fontsize=12, ha='left', va='top', style='italic')

        y_pos = 0.88
        for color_idx, symbol in symbol_map.items():
            dmc_color, dmc_name = self.color_mapping[color_idx]

            ax.text(0.1, y_pos, symbol, fontsize=14, ha='left', va='center',
                    fontfamily='DejaVu Sans', fontweight='bold')

            ax.text(0.2, y_pos, dmc_name, fontsize=11, ha='left', va='center',
                    fontweight='bold')

            rect = patches.Rectangle((0.5, y_pos - 0.015), 0.1, 0.03,
                                     facecolor=np.array(dmc_color) / 255,
                                     edgecolor='black', linewidth=1)
            ax.add_patch(rect)

            rgb_text = f"RGB({dmc_color[0]}, {dmc_color[1]}, {dmc_color[2]})"
            ax.text(0.65, y_pos, rgb_text, fontsize=9, ha='left', va='center',
                    alpha=0.7, style='italic')

            y_pos -= 0.04

            if y_pos < 0.05:
                ax.text(0.1, 0.02, '... (продолжение на следующей странице)',
                        fontsize=10, style='italic', alpha=0.7)
                break

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Легенда сохранена: {output_path}")

    #Основной процесс
    def convert(self, image_path, max_colors=20, max_stitches=100, output_dir="cross_stitch_output"):
        # Создание директории
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # Последовательность обработки
        try:
            self.load_image(image_path) # 1. Загрузка

            w, h = self.resize_image(max_stitches) # 2. Изменение размера

            self.reduce_colors(max_colors) # 3. Уменьшение цветов

            pattern_file = os.path.join(output_dir, f"{base_name}_pattern.png")
            symbol_map = self.create_stitch_chart(pattern_file) # 4. Создание схемы

            legend_file = os.path.join(output_dir, f"{base_name}_legend.png")
            self.create_legend(symbol_map, legend_file)# 5. Легенда

            print(f"Размер схемы: {w}×{h} крестиков")
            print(f"Использовано цветов: {self.actual_colors}")
            print(f"Результаты в папке: {output_dir}")
            print(f"   • {os.path.basename(pattern_file)} - схема с символами")
            print(f"   • {os.path.basename(legend_file)} - легенда цветов")

            return True

        except Exception as e:
            print(f"Ошибка: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Конвертер изображений в схемы для вышивки крестиком')
    parser.add_argument('image', help='Путь к изображению')
    parser.add_argument('--max-colors', type=int, default=20,
                        help='Максимальное количество цветов ниток (по умолчанию: 20)')
    parser.add_argument('--max-stitches', type=int, default=100,
                        help='Максимальное количество крестиков (по умолчанию: 100)')
    parser.add_argument('--output-dir', default='cross_stitch_output',
                        help='Папка для результатов (по умолчанию: cross_stitch_output)')

    args = parser.parse_args()

    converter = CrossStitchConverter()
    success = converter.convert(
        image_path=args.image,
        max_colors=args.max_colors,
        max_stitches=args.max_stitches,
        output_dir=args.output_dir
    )

    if not success:
        print("Конвертация не удалась")


if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        image_files = []
        for file in os.listdir('.'):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image_files.append(file)

        if image_files:
            image_path = image_files[0]

            converter = CrossStitchConverter()
            converter.convert(
                image_path=image_path,
                max_colors=20,
                max_stitches=80,
                output_dir="cross_stitch_output"
            )
        else:
            print("В папке нет изображений!")
    else:
        main()