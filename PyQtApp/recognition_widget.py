from PyQt6.QtWidgets import QWidget, QDockWidget, QApplication, QLayout
from PyQt6.QtWidgets import QDialog, QCheckBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import pyqtSlot, Qt
import numpy as np
import cv2


class RecognitionWidget(QDockWidget, QWidget):
    def __init__(self, window_name, map_list: list):
        super().__init__()
        self.setWindowTitle(window_name)
        self.map_list = map_list
        self.setWidget(QWidget(self))
        self.main_layout = QVBoxLayout()
        self.widget().setLayout(self.main_layout)
        # self.setFixedSize(600, 600)
        # self.setMinimumSize(200, 200)  # Минимальный размер окна
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Позволяет растягиваться

        self.img_width, self.img_height = 320, 320
        self.drons_btns = {}

        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.img_frame = QLabel()
        pixmap = QPixmap('assets/spectrum_img.png')
        self.img_frame.setPixmap(pixmap)
        self.img_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Центрирование изображения
        # self.img_frame.setMaximumSize(pixmap.width(), pixmap.height())
        self.img_frame.setScaledContents(True)      # масштабирование изображения по размеру виджета

        for name in self.map_list:
            drone_btn = QPushButton(name)
            self.drons_btns[name] = drone_btn

    def add_widgets_to_layout(self):
        img_frame_layout = QHBoxLayout()
        img_frame_layout.addWidget(self.img_frame)
        self.main_layout.addLayout(img_frame_layout)

        drons_btns_layout = QHBoxLayout()
        for btn in self.drons_btns.values():
            drons_btns_layout.addWidget(btn)
        self.main_layout.addLayout(drons_btns_layout)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        # rotate = self.rotate_cb.currentData()
        # if rotate is not None:
            # rgb_image = cv2.rotate(rgb_image, rotate)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data,
                                      w,
                                      h,
                                      bytes_per_line,
                                      QImage.Format.Format_RGB888)
        picture = convert_to_Qt_format.scaled(self.img_width,  self.img_height, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(picture)

    @pyqtSlot(dict)
    def update_state(self, info: dict):
        band_name = info['band_name']
        for drone_dict in info['drones']:
            if drone_dict['state']:
                self.drons_btns[drone_dict['name']].setStyleSheet("background-color: #F0483C; "
                                                                  "font: bold; "
                                                                  "color: #FFFFFF")
            else:
                self.drons_btns[drone_dict['name']].setStyleSheet("background-color: ")

        if 'img' in info:
            img_arr = np.frombuffer(info['img'], dtype=np.uint8).reshape((640, 640, 3))
            img_pixmap = self.convert_cv_qt(img_arr)
            self.img_frame.setPixmap(img_pixmap)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    main_window = RecognitionWidget(window_name='afasd', map_list=['1', '2', '3', '4'])
    main_window.show()
    sys.exit(app.exec())
