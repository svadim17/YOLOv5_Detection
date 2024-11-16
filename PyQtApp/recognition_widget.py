from PyQt6.QtWidgets import QWidget, QDockWidget, QApplication, QLayout, QSlider, QSpinBox
from PyQt6.QtWidgets import QDialog, QCheckBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt
import numpy as np
import cv2


class RecognitionWidget(QDockWidget, QWidget):
    signal_zscale_changed = pyqtSignal(str, int, int)

    def __init__(self, window_name, map_list: list, zscale_settings: list):
        super().__init__()
        self.name = window_name
        self.map_list = map_list
        self.zscale_settings = zscale_settings

        self.setWindowTitle(self.name)

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
        self.img_frame.setPixmap(pixmap.scaled(256, 256))
        self.img_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Центрирование изображения
        # self.img_frame.setMaximumSize(pixmap.width(), pixmap.height())
        self.img_frame.setScaledContents(True)      # масштабирование изображения по размеру виджета

        self.slider_zscale_max = QSlider()
        self.slider_zscale_max.setOrientation(Qt.Orientation.Vertical)
        self.slider_zscale_max.setRange(-120, 120)
        self.slider_zscale_max.setSingleStep(1)
        self.slider_zscale_max.setValue(self.zscale_settings[1])
        self.slider_zscale_max.sliderReleased.connect(lambda: self.slider_zmax_changed(self.slider_zscale_max.value()))
        self.spb_slider_zscale_max = QSpinBox()
        self.spb_slider_zscale_max.setRange(-120, 120)
        self.spb_slider_zscale_max.setValue(self.slider_zscale_max.value())
        self.spb_slider_zscale_max.valueChanged.connect(self.slider_zmax_changed)

        self.slider_zscale_min = QSlider()
        self.slider_zscale_min.setOrientation(Qt.Orientation.Vertical)
        self.slider_zscale_min.setRange(-120, 120)
        self.slider_zscale_min.setSingleStep(1)
        self.slider_zscale_min.setValue(self.zscale_settings[0])
        self.slider_zscale_min.sliderReleased.connect(lambda: self.slider_zmin_changed(self.slider_zscale_min.value()))
        self.spb_slider_zscale_min = QSpinBox()
        self.spb_slider_zscale_min.setRange(-120, 120)
        self.spb_slider_zscale_min.setValue(self.slider_zscale_min.value())
        self.spb_slider_zscale_min.valueChanged.connect(self.slider_zmin_changed)

        for name in self.map_list:
            drone_btn = QPushButton(name)
            self.drons_btns[name] = drone_btn

    def add_widgets_to_layout(self):
        slider_zmax_layout = QVBoxLayout()
        slider_zmax_layout.addWidget(self.slider_zscale_max)
        slider_zmax_layout.addWidget(self.spb_slider_zscale_max)

        slider_zmin_layout = QVBoxLayout()
        slider_zmin_layout.addWidget(self.spb_slider_zscale_min)
        slider_zmin_layout.addWidget(self.slider_zscale_min)

        sliders_layout = QVBoxLayout()
        sliders_layout.addLayout(slider_zmax_layout)
        sliders_layout.addLayout(slider_zmin_layout)

        img_frame_layout = QHBoxLayout()
        img_frame_layout.addWidget(self.img_frame)
        img_frame_layout.addLayout(sliders_layout)

        drons_btns_layout = QHBoxLayout()
        for btn in self.drons_btns.values():
            drons_btns_layout.addWidget(btn)

        self.main_layout.addLayout(img_frame_layout)
        self.main_layout.addLayout(drons_btns_layout)

    def slider_zmax_changed(self, value: int):
        self.slider_zscale_max.setValue(value)
        self.spb_slider_zscale_max.setValue(value)
        self.signal_zscale_changed.emit(self.name, self.slider_zscale_min.value(), self.slider_zscale_max.value())

    def slider_zmin_changed(self, value: int):
        self.slider_zscale_min.setValue(value)
        self.spb_slider_zscale_min.setValue(value)
        self.signal_zscale_changed.emit(self.name, self.slider_zscale_min.value(), self.slider_zscale_max.value())

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        # rgb_image = cv2.resize(rgb_image, (self.img_width, self.img_height))
        # rotate = self.rotate_cb.currentData()
        # if rotate is not None:
            # rgb_image = cv2.rotate(rgb_image, rotate)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        picture = QImage(rgb_image.data,
                                      w,
                                      h,
                                      bytes_per_line,
                                      QImage.Format.Format_RGB888)
        if self.img_frame.width() < w and self.img_frame.height() < h:
            picture = picture.scaled(self.img_width,  self.img_height)
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

    # def resizeEvent(self, a0):
    #     self.img_width = self.img_frame.width()
    #     self.img_height = self.img_frame.height()
        # self.img_frame.setFixedWidth(self.img_width)
        # self.img_frame.setFixedHeight(self.img_height)
        #
        # print(self.img_width, self.img_height)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    main_window = RecognitionWidget(window_name='afasd', map_list=['1', '2', '3', '4'])
    main_window.show()
    sys.exit(app.exec())
