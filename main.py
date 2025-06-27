import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                            QComboBox, QLineEdit, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Обработчик изображений с PyTorch")
        self.setGeometry(100, 100, 1000, 600)
        
        self.image = None
        self.processed_image = None
        self.tensor_image = None
        
        # Преобразования для PyTorch
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.init_ui()
        
    def init_ui(self):
        # Главный виджет и layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)
        
        # Метки для отображения изображений
        self.original_label = QLabel("Исходное изображение")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 300)
        
        self.processed_label = QLabel("Обработанное изображение")
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setMinimumSize(400, 300)
        
        # Layout для изображений
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.original_label)
        image_layout.addWidget(self.processed_label)
        layout.addLayout(image_layout)
        
        # Кнопки загрузки изображений
        load_buttons_layout = QHBoxLayout()
        
        self.load_button = QPushButton("Загрузить изображение")
        self.load_button.clicked.connect(self.load_image)
        load_buttons_layout.addWidget(self.load_button)
        
        self.camera_button = QPushButton("Сделать фото")
        self.camera_button.clicked.connect(self.take_photo)
        load_buttons_layout.addWidget(self.camera_button)
        
        layout.addLayout(load_buttons_layout)
        
        # Опции обработки
        self.channel_combo = QComboBox()
        self.channel_combo.addItems([
            "Красный канал", 
            "Зеленый канал", 
            "Синий канал", 
            "Негатив", 
            "Размытие по Гауссу", 
            "Нарисовать круг"
        ])
        layout.addWidget(self.channel_combo)
        
        # Параметры для размытия
        self.blur_widget = QWidget()
        self.blur_layout = QHBoxLayout(self.blur_widget)
        self.blur_label = QLabel("Размер ядра (нечетное число):")
        self.blur_input = QLineEdit()
        self.blur_input.setPlaceholderText("например, 5")
        self.blur_layout.addWidget(self.blur_label)
        self.blur_layout.addWidget(self.blur_input)
        self.blur_widget.setVisible(False)
        layout.addWidget(self.blur_widget)
        
        # Параметры для круга
        self.circle_widget = QWidget()
        self.circle_layout = QHBoxLayout(self.circle_widget)
        self.circle_x_label = QLabel("X:")
        self.circle_x_input = QLineEdit()
        self.circle_x_input.setPlaceholderText("Координата X")
        
        self.circle_y_label = QLabel("Y:")
        self.circle_y_input = QLineEdit()
        self.circle_y_input.setPlaceholderText("Координата Y")
        
        self.circle_r_label = QLabel("Радиус:")
        self.circle_r_input = QLineEdit()
        self.circle_r_input.setPlaceholderText("Радиус круга")
        
        self.circle_layout.addWidget(self.circle_x_label)
        self.circle_layout.addWidget(self.circle_x_input)
        self.circle_layout.addWidget(self.circle_y_label)
        self.circle_layout.addWidget(self.circle_y_input)
        self.circle_layout.addWidget(self.circle_r_label)
        self.circle_layout.addWidget(self.circle_r_input)
        self.circle_widget.setVisible(False)
        layout.addWidget(self.circle_widget)
        
        # Кнопка применения изменений
        self.apply_button = QPushButton("Применить изменения")
        self.apply_button.clicked.connect(self.process_image)
        layout.addWidget(self.apply_button)
        
        # Обработчик изменения выбора в комбобоксе
        self.channel_combo.currentTextChanged.connect(self.update_ui)
        
    def update_ui(self, text):
        self.blur_widget.setVisible(text == "Размытие по Гауссу")
        self.circle_widget.setVisible(text == "Нарисовать круг")
        
    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Открыть изображение", "", 
                                                "Изображения (*.png *.jpg *.jpeg)", options=options)
        if file_name:
            try:
                # Загрузка изображения с помощью OpenCV 
                self.image = cv2.imread(file_name)
                if self.image is None:
                    raise ValueError("Не удалось прочитать файл изображения.")
                
                # Конвертация в PyTorch tensor
                pil_image = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
                self.tensor_image = self.transform(pil_image).unsqueeze(0)
                
                self.display_image(self.image, self.original_label)
                self.show_message("Успех", "Изображение успешно загружено!")
            except Exception as e:
                self.show_message("Ошибка", f"Ошибка загрузки изображения: {str(e)}")
    
    def take_photo(self):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.show_message("Ошибка", "Не удалось подключиться к веб-камере. Возможные решения:\n"
                                          "1. Проверьте подключение камеры\n"
                                          "2. Предоставьте приложению разрешение на использование камеры\n"
                                          "3. Попробуйте другую камеру\n"
                                          "4. Перезапустите приложение")
                return
            
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                self.image = frame
                # Конвертация в PyTorch tensor 
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.tensor_image = self.transform(pil_image).unsqueeze(0)
                
                self.display_image(self.image, self.original_label)
                self.show_message("Успех", "Фото успешно сделано!")
            else:
                self.show_message("Ошибка", "Не удалось получить изображение с камеры")
        except Exception as e:
            self.show_message("Ошибка", f"Ошибка работы с камерой: {str(e)}")
    
    def process_image(self):
        if self.image is None or self.tensor_image is None:
            return
            
        option = self.channel_combo.currentText()
        
        try:
            if option == "Красный канал":
                # Создаем маску для красного канала 
                mask = torch.zeros_like(self.tensor_image)
                mask[:, 0, :, :] = 1  # Оставляем только красный канал
                result = (self.tensor_image * mask).squeeze().permute(1, 2, 0).numpy() * 255
                self.processed_image = result.astype(np.uint8)
                
            elif option == "Зеленый канал":
                # Маска для зеленого канала 
                mask = torch.zeros_like(self.tensor_image)
                mask[:, 1, :, :] = 1  # Оставляем только зеленый канал
                result = (self.tensor_image * mask).squeeze().permute(1, 2, 0).numpy() * 255
                self.processed_image = result.astype(np.uint8)
                
            elif option == "Синий канал":
                # Маска для синего канала 
                mask = torch.zeros_like(self.tensor_image)
                mask[:, 2, :, :] = 1  # Оставляем только синий канал
                result = (self.tensor_image * mask).squeeze().permute(1, 2, 0).numpy() * 255
                self.processed_image = result.astype(np.uint8)
                
            elif option == "Негатив":
                # Реализация негатива через PyTorch
                result = (1.0 - self.tensor_image).squeeze().permute(1, 2, 0).numpy() * 255
                self.processed_image = result.astype(np.uint8)
                
            elif option == "Размытие по Гауссу":
                kernel_size = self.blur_input.text()
                if kernel_size:
                    try:
                        ksize = int(kernel_size)
                        if ksize > 0 and ksize % 2 == 1:
                            # Используем OpenCV для размытия 
                            self.processed_image = cv2.GaussianBlur(self.image, (ksize, ksize), 0)
                        else:
                            self.show_message("Предупреждение", "Размер ядра должен быть положительным нечетным числом")
                            return
                    except ValueError:
                        return
                else:
                    return
                    
            elif option == "Нарисовать круг":
                x = self.circle_x_input.text()
                y = self.circle_y_input.text()
                r = self.circle_r_input.text()
                
                if x and y and r:
                    try:
                        x = int(x)
                        y = int(y)
                        r = int(r)
                        
                        if x >= 0 and y >= 0 and r > 0:
                            # Создаем копию изображения (
                            self.processed_image = self.image.copy()
                            # Рисуем красный круг (BGR: (0,0,255))
                            cv2.circle(self.processed_image, (x, y), r, (0, 0, 255), 2)
                        else:
                            self.show_message("Предупреждение", "Координаты должны быть положительными, а радиус больше 0")
                            return
                    except ValueError:
                        return
                else:
                    return
            
            # Для операций с цветовыми каналами и негатива конвертируем RGB в BGR
            if option in ["Красный канал", "Зеленый канал", "Синий канал", "Негатив"]:
                self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
            
            self.display_image(self.processed_image, self.processed_label)
        except Exception as e:
            self.show_message("Ошибка", f"Ошибка обработки изображения: {str(e)}")
    
    def display_image(self, image, label):
        if image is not None:
            # Конвертация BGR в RGB для отображения в Qt
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio))
    
    def show_message(self, title, message):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec_())