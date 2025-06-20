from PyQt6.QtWidgets import (QWidget, QDockWidget, QApplication, QSpinBox, QDoubleSpinBox,
                             QAbstractSpinBox, QDialog, QCheckBox, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QSpacerItem, QSizePolicy, QGroupBox, QComboBox, QToolTip)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import pyqtSlot, Qt, pyqtSignal, QPoint
import numpy as np
import cv2
import qdarktheme
from .gRPC_thread import ChannelInfo


class RecognitionOptions(QWidget):
    signal_recogn_settings = pyqtSignal(str, int, float, float)
    signal_zscale_changed = pyqtSignal(str, int, int)
    signal_gain_changed = pyqtSignal(str, int)

    def __init__(self, name: str, zscale_settings: list, current_recogn_settings: dict, channel_info: ChannelInfo):
        super().__init__()
        self.name = name
        self.zscale_settings = zscale_settings
        self.accum_size = current_recogn_settings['accum_size']
        self.threshold = current_recogn_settings['threshold']
        self.exceedance = current_recogn_settings['exceedance']
        self.channel_info = channel_info

        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle(name)
        self.setFixedSize(300, 500)

        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(15)
        self.setLayout(self.main_layout)
        self.create_widgets()
        self.link_events()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.box_decision = QGroupBox('Decision')

        self.l_slider_threshold = QLabel('Threshold')
        self.slider_threshold = QSlider()
        self.slider_threshold.setRange(5, 100)
        self.slider_threshold.setSingleStep(10)
        self.slider_threshold.setOrientation(Qt.Orientation.Horizontal)
        self.slider_threshold.setValue(int(self.threshold * 100))
        self.slider_threshold.setToolTip(str(self.slider_threshold.value()))
        self.spb_slider_threshold = QDoubleSpinBox()
        self.spb_slider_threshold.setRange(0.05, 1)
        self.spb_slider_threshold.setSingleStep(0.05)
        self.spb_slider_threshold.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spb_slider_threshold.setValue(self.threshold)

        self.l_spb_accum_size = QLabel('Accumulation size')
        self.spb_accum_size = QSpinBox()
        self.spb_accum_size.setRange(1, 30)
        self.spb_accum_size.setSingleStep(1)
        self.spb_accum_size.setValue(self.accum_size)

        self.l_exceedance = QLabel('Exceedance')
        self.spb_exceedance = QDoubleSpinBox()
        self.spb_exceedance.setRange(0.05, 1)
        self.spb_exceedance.setSingleStep(0.05)
        self.spb_exceedance.setValue(self.exceedance)

        self.box_zscale = QGroupBox('Z-scale settings')

        self.slider_zscale_max = QSlider()
        self.slider_zscale_max.setOrientation(Qt.Orientation.Vertical)
        self.slider_zscale_max.setRange(-120, 120)
        self.slider_zscale_max.setSingleStep(1)
        self.slider_zscale_max.setValue(self.zscale_settings[1])
        self.slider_zscale_max.sliderReleased.connect(lambda: self.slider_zmax_changed(self.slider_zscale_max.value()))
        self.spb_slider_zscale_max = QSpinBox()
        self.spb_slider_zscale_max.setMaximumWidth(60)
        self.spb_slider_zscale_max.setRange(-120, 120)
        self.spb_slider_zscale_max.setValue(self.slider_zscale_max.value())
        self.spb_slider_zscale_max.valueChanged.connect(self.slider_zmax_changed)
        self.l_slider_zscale_max = QLabel('Z-Max')

        self.slider_zscale_min = QSlider()
        self.slider_zscale_min.setOrientation(Qt.Orientation.Vertical)
        self.slider_zscale_min.setRange(-120, 120)
        self.slider_zscale_min.setSingleStep(1)
        self.slider_zscale_min.setValue(self.zscale_settings[0])
        self.slider_zscale_min.sliderReleased.connect(lambda: self.slider_zmin_changed(self.slider_zscale_min.value()))
        self.spb_slider_zscale_min = QSpinBox()
        self.spb_slider_zscale_min.setMaximumWidth(60)
        self.spb_slider_zscale_min.setRange(-120, 120)
        self.spb_slider_zscale_min.setValue(self.slider_zscale_min.value())
        self.spb_slider_zscale_min.valueChanged.connect(self.slider_zmin_changed)
        self.l_slider_zscale_min = QLabel('Z-Min')

    def link_events(self):
        self.slider_threshold.valueChanged.connect(lambda: self.slider_threshold_value_changed(
                                                     self.slider_threshold.value() / 100))
        self.slider_threshold.sliderReleased.connect(self.recognition_settings_changed)
        self.spb_slider_threshold.valueChanged.connect(self.slider_threshold_value_changed)
        self.spb_slider_threshold.valueChanged.connect(self.recognition_settings_changed)
        self.spb_accum_size.valueChanged.connect(self.recognition_settings_changed)
        self.spb_exceedance.valueChanged.connect(self.recognition_settings_changed)

    def add_widgets_to_layout(self):
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        threshold_slider_layout = QHBoxLayout()
        threshold_slider_layout.addWidget(self.slider_threshold)
        threshold_slider_layout.addWidget(self.spb_slider_threshold)
        threshold_layout = QVBoxLayout()
        threshold_layout.setSpacing(5)
        threshold_layout.addWidget(self.l_slider_threshold)
        threshold_layout.addLayout(threshold_slider_layout)

        accumulation_layout = QVBoxLayout()
        accumulation_layout.setSpacing(5)
        accumulation_layout.addWidget(self.l_spb_accum_size)
        accumulation_layout.addWidget(self.spb_accum_size)

        exceedance_layout = QVBoxLayout()
        exceedance_layout.setSpacing(5)
        exceedance_layout.addWidget(self.l_exceedance)
        exceedance_layout.addWidget(self.spb_exceedance)

        accum_and_exceed_layout = QHBoxLayout()
        accum_and_exceed_layout.addLayout(accumulation_layout)
        accum_and_exceed_layout.addLayout(exceedance_layout)

        slider_zmax_layout = QVBoxLayout()
        slider_zmax_layout.addWidget(self.slider_zscale_max)
        slider_zmax_layout.addWidget(self.spb_slider_zscale_max)
        slider_zmax_layout.addWidget(self.l_slider_zscale_max)

        slider_zmin_layout = QVBoxLayout()
        slider_zmin_layout.addWidget(self.slider_zscale_min)
        slider_zmin_layout.addWidget(self.spb_slider_zscale_min)
        slider_zmin_layout.addWidget(self.l_slider_zscale_min)

        sliders_layout = QHBoxLayout()
        sliders_layout.addLayout(slider_zmin_layout)
        sliders_layout.addLayout(slider_zmax_layout)

        cntrls_layout = QVBoxLayout()
        cntrls_layout.setSpacing(30)
        cntrls_layout.addLayout(threshold_layout)
        cntrls_layout.addLayout(accum_and_exceed_layout)

        self.box_decision.setLayout(cntrls_layout)
        self.box_zscale.setLayout(sliders_layout)

        self.main_layout.addWidget(self.box_decision)
        self.main_layout.addWidget(self.box_zscale)

    @pyqtSlot(float)
    def slider_threshold_value_changed(self, value: float):
        self.slider_threshold.setValue(int(value * 100))
        # self.slider_threshold.setToolTip(str(value))
        self.update_slider_tooltip(slider=self.slider_threshold, value=value)
        # self.spb_slider_threshold.setValue(value)

    def recognition_settings_changed(self):
        self.spb_slider_threshold.setValue(self.slider_threshold.value() / 100)
        self.signal_recogn_settings.emit(self.name,
                                         self.spb_accum_size.value(),
                                         self.spb_slider_threshold.value(),
                                         self.spb_exceedance.value())

    @pyqtSlot(int)
    def slider_zmax_changed(self, value: int):
        self.slider_zscale_max.setValue(value)
        self.spb_slider_zscale_max.setValue(value)
        self.signal_zscale_changed.emit(self.name, self.slider_zscale_min.value(), self.slider_zscale_max.value())

    @pyqtSlot(int)
    def slider_zmin_changed(self, value: int):
        self.slider_zscale_min.setValue(value)
        self.spb_slider_zscale_min.setValue(value)
        self.signal_zscale_changed.emit(self.name, self.slider_zscale_min.value(), self.slider_zscale_max.value())

    def update_slider_tooltip(self, slider, value):
        # Получаем позицию слайдера в глобальных координатах
        slider_pos = slider.mapToGlobal(slider.pos())

        # Вычисляем позицию "ползунка" (thumb) слайдера
        thumb_width = slider.style().pixelMetric(
            slider.style().PixelMetric.PM_SliderControlThickness
        )
        thumb_pos = int(
            (value - slider.minimum()) / (slider.maximum() - slider.minimum())
            * (slider.width() - thumb_width)
        )

        # Позиция для ToolTip (чуть выше ползунка)
        tooltip_x = slider_pos.x() + thumb_pos
        tooltip_y = slider_pos.y() - 20  # Смещение вверх

        # Показываем ToolTip с текущим значением
        QToolTip.showText(
            slider.mapToGlobal(slider.rect().topLeft() + QPoint(thumb_pos, -20)),
            str(value),
            slider
        )


if __name__ == '__main__':
    app = QApplication([])
    qdarktheme.setup_theme()
    window = RecognitionOptions(name='test',
                                zscale_settings=[10, 30],
                                current_recogn_settings={'accum_size': 10, 'threshold': 0.5, 'exceedance': 0.7})
    window.show()
    app.exec()
