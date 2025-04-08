import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QSlider, QVBoxLayout, QWidget, QToolTip
from PyQt6.QtCore import Qt, QPoint



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Slider with Persistent ToolTip")
        self.setGeometry(100, 100, 400, 200)

        # Создаем центральный виджет и layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Создаем слайдер
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        layout.addWidget(self.slider)

        # Подключаем сигнал valueChanged к обработчику
        self.slider.valueChanged.connect(self.update_tooltip)

    def update_tooltip(self, value):
        # Получаем позицию слайдера в глобальных координатах
        slider_pos = self.slider.mapToGlobal(self.slider.pos())

        # Вычисляем позицию "ползунка" (thumb) слайдера
        thumb_width = self.slider.style().pixelMetric(
            self.slider.style().PixelMetric.PM_SliderControlThickness
        )
        thumb_pos = int(
            (value - self.slider.minimum()) / (self.slider.maximum() - self.slider.minimum())
            * (self.slider.width() - thumb_width)
        )

        # Позиция для ToolTip (чуть выше ползунка)
        tooltip_x = slider_pos.x() + thumb_pos
        tooltip_y = slider_pos.y() - 20  # Смещение вверх

        # Показываем ToolTip с текущим значением
        QToolTip.showText(
            self.slider.mapToGlobal(self.slider.rect().topLeft() + QPoint(thumb_pos, -20)),
            str(value),
            self.slider
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())