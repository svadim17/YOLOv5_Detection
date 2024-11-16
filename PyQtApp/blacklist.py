from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QVBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QColor

class DoubleSlider(QWidget):
    def __init__(self):
        super(DoubleSlider, self).__init__()

        self.setWindowTitle('Двухползунковый слайдер')
        self.setGeometry(200, 200, 400, 100)

        # Значения для ползунков
        self.min_value = 20
        self.max_value = 80

        # Диапазон слайдера
        self.min_range = 0
        self.max_range = 100

        self.setMinimumWidth(400)
        self.setMinimumHeight(60)

        # Метка для отображения значений ползунков
        self.label = QLabel(f"Min: {self.min_value}, Max: {self.max_value}", self)

        # Основной вертикальный layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    def paintEvent(self, event):
        # Рисуем слайдер и два ползунка
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Рисуем основной слайдер
        slider_rect = QRect(50, 40, self.width() - 100, 10)
        painter.setBrush(QColor(220, 220, 220))  # Светло-серый цвет для фона слайдера
        painter.drawRect(slider_rect)

        # Вычисляем позиции ползунков на слайдере
        min_pos = (self.min_value - self.min_range) / (self.max_range - self.min_range) * (slider_rect.width())
        max_pos = (self.max_value - self.min_range) / (self.max_range - self.min_range) * (slider_rect.width())

        # Рисуем ползунки
        min_thumb = QPoint(slider_rect.left() + min_pos, slider_rect.top() + slider_rect.height() / 2)
        max_thumb = QPoint(slider_rect.left() + max_pos, slider_rect.top() + slider_rect.height() / 2)
        painter.setBrush(QColor(50, 150, 250))  # Синий цвет для ползунков
        painter.drawEllipse(min_thumb, 8, 8)
        painter.drawEllipse(max_thumb, 8, 8)

    def mousePressEvent(self, event):
        # Логика для перетаскивания ползунков
        slider_rect = QRect(50, 40, self.width() - 100, 10)
        min_pos = (self.min_value - self.min_range) / (self.max_range - self.min_range) * (slider_rect.width())
        max_pos = (self.max_value - self.min_range) / (self.max_range - self.min_range) * (slider_rect.width())
        min_thumb = QPoint(slider_rect.left() + min_pos, slider_rect.top() + slider_rect.height() / 2)
        max_thumb = QPoint(slider_rect.left() + max_pos, slider_rect.top() + slider_rect.height() / 2)

        # Проверяем, был ли клик по ползункам
        if min_thumb.x() - 8 <= event.x() <= min_thumb.x() + 8 and min_thumb.y() - 8 <= event.y() <= min_thumb.y() + 8:
            self.dragging_min = True
            self.dragging_max = False
        elif max_thumb.x() - 8 <= event.x() <= max_thumb.x() + 8 and max_thumb.y() - 8 <= event.y() <= max_thumb.y() + 8:
            self.dragging_max = True
            self.dragging_min = False
        else:
            self.dragging_min = False
            self.dragging_max = False

    def mouseMoveEvent(self, event):
        # Двигаем ползунки в зависимости от клика
        if self.dragging_min or self.dragging_max:
            slider_rect = QRect(50, 40, self.width() - 100, 10)
            mouse_pos = event.x() - slider_rect.left()

            # Ограничаем движение ползунков в пределах слайдера
            if self.dragging_min:
                new_min_value = min(self.max_range, max(self.min_range, mouse_pos / slider_rect.width() * (self.max_range - self.min_range)))
                if new_min_value < self.max_value:
                    self.min_value = new_min_value
            elif self.dragging_max:
                new_max_value = max(self.min_range, min(self.max_range, mouse_pos / slider_rect.width() * (self.max_range - self.min_range)))
                if new_max_value > self.min_value:
                    self.max_value = new_max_value

            self.update()  # Перерисовываем слайдер

        # Обновляем метку с новыми значениями
        self.label.setText(f"Min: {int(self.min_value)}, Max: {int(self.max_value)}")

    def mouseReleaseEvent(self, event):
        # Завершаем перетаскивание
        self.dragging_min = False
        self.dragging_max = False


if __name__ == "__main__":
    app = QApplication([])
    window = DoubleSlider()
    window.show()
    app.exec()