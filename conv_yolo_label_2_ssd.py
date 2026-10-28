import os
import glob

def yolo_to_corners(xc, yc, w, h, img_width, img_height):
    """
    Конвертирует нормализованные координаты YOLO в пиксельные координаты боксов.
    Возвращает: x1min, y1min, x2max, y2max
    """
    # Перевод в пиксели
    x_center_px = xc * img_width
    y_center_px = yc * img_height
    width_px = w * img_width
    height_px = h * img_height

    # Вычисление боксов (левый верхний и правый нижний углы)
    x1 = x_center_px - (width_px / 2)
    y1 = y_center_px - (height_px / 2)
    x2 = x_center_px + (width_px / 2)
    y2 = y_center_px + (height_px / 2)

    # Округление
    
    return round(x1), round(y1), round(x2), round(y2)

def convert_annotations(input_dir, output_dir, img_width, img_height):
    # Получаем список всех .txt файлов
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))

    if not txt_files:
        print(f"Файлы .txt не найдены в папке {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    for txt_path in txt_files:
        filename = os.path.basename(txt_path)
        output_path = os.path.join(output_dir, filename)

        with open(txt_path, 'r') as f_in, open(output_path, 'w') as f_out:
            lines = f_in.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue # Пропускаем некорректные строки

                class_id = parts[0]
                xc, yc, w, h = map(float, parts[1:])

                x1, y1, x2, y2 = yolo_to_corners(xc, yc, w, h, img_width, img_height)

                # Форматируем вывод: class_id, x1, y1, x2, y2
                
                f_out.write(f"{(class_id)}, {x1}, {y1}, {x2}, {y2}\n")


if __name__ == "__main__":
    # ================= НАСТРОЙКИ =================
    # Укажите пути к папкам
    INPUT_DIR = './val'   # Папка с исходными .txt файлами
    OUTPUT_DIR = './test' # Папка для результатов

    # ВАЖНО: Укажите реальные размеры изображений в пикселях
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 640
    # =============================================
    convert_annotations(INPUT_DIR, OUTPUT_DIR, IMAGE_WIDTH, IMAGE_HEIGHT)
    print("Готово!")
