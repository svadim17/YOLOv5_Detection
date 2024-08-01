import time
import cv2
import pandas
import pyqtgraph
import yaml
from PyQt5 import QtWidgets, Qt, QtCore
from PyQt5.QtWidgets import (QApplication, QToolBar, QAction, QDockWidget, QWidget, QLabel,
                             QVBoxLayout, QHBoxLayout, QSplitter, QPushButton, QDialog,
                             QLineEdit, QFileDialog, QSpinBox, QCheckBox, QComboBox)
from PyQt5.QtGui import QPixmap, QImage
from loguru import logger
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QSize
from multiprocessing import Process, Manager, Queue
import numpy as np
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
import pandas as pd
# import Alinx_DualPort as nn
import LV_to_Python as nn
import datetime
import yaml
from collections import deque


GLOBAL_COLORS = {'noise': (220, 138, 221),
                 'dji': (255, 163, 72),
                 'wifi': (0, 255, 12),
                 'autel': (165, 29, 45),
                 'autel_lite': (129, 61, 156),
                 'autel_max_4n(t)': (198, 70, 0),
                 'autel_tag': (26, 95, 180),
                 'fpv': (98, 160, 234),
                 '3G/4G': (255, 255, 255),
                 }



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # self.resize(1024, 600)
        self.connections = []
        # self.recogn_widget = RecognitionViewerWidget('127.0.0.1', 10091, window_name='Default')
        # self.addDockWidget(Qt.Qt.RightDockWidgetArea, self.recogn_widget)
        # self.connections.append(self.recogn_widget)

        self.config_path = None

        self.histogram_dock_widget = None
        self.adding_window = AddingConnectionWindow()
        self.create_toolbar_actions()
        self.init_toolbar()
        self.adding_window.signal_new_connection.connect(self.add_new_connection)

    def create_toolbar_actions(self):
        self.start_action = QAction('Start all')
        self.start_action.triggered.connect(self.start_all)

        self.add_action = QAction('Add connection')
        self.add_action.triggered.connect(self.open_adding_window)

        self.save_config_action = QAction('Save config')
        self.save_config_action.triggered.connect(self.save_config)

        self.load_config_action = QAction('Load config')
        self.load_config_action.triggered.connect(self.load_config)

        self.exit_action = QAction('Exit')
        self.exit_action.triggered.connect(self.close)

    def init_toolbar(self):
        self.toolbar = QToolBar('Main')
        self.addToolBar(Qt.Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        self.toolbar.addAction(self.start_action)
        self.toolbar.addAction(self.add_action)
        self.toolbar.addAction(self.save_config_action)
        self.toolbar.addAction(self.load_config_action)
        self.toolbar.addAction(self.exit_action)

    def open_adding_window(self):
        self.adding_window.btn_add_connection.setDisabled(True)
        self.adding_window.show()

    def add_new_connection(self, info: dict):
        new_connection = RecognitionViewerWidget(ip=info['ip'],
                                                 port=info['port'],
                                                 window_name=info['name'],
                                                 weights_path=info['weights_path'])

        # new_connection.visibilityChanged.connect(lambda: self.handle_visibility_change(new_connection))
        self.addDockWidget(Qt.Qt.LeftDockWidgetArea, new_connection)
        try:
            self.tabifyDockWidget(self.connections[0], new_connection)
        except: pass
        self.connections.append(new_connection)

        if self.histogram_dock_widget is None:
            self.histogram_dock_widget = HistogramViewerWidget(new_connection.name)
            self.addDockWidget(Qt.Qt.LeftDockWidgetArea, self.histogram_dock_widget)
            self.tabifyDockWidget(self.connections[0], self.histogram_dock_widget)
        else:
            self.histogram_dock_widget.create_histogram(new_connection.name)
        new_connection.dataThread.signal_recogn_df.connect(self.histogram_dock_widget.update_data)

    def load_config(self):
        self.config_path, _ = QFileDialog.getOpenFileName(filter='*.yaml')
        try:
            with open(self.config_path, encoding='utf-8') as f:
                self.config = dict(yaml.load(f, Loader=yaml.SafeLoader))
                for conn in self.config['connections']:
                    self.add_new_connection(info=conn)
            logger.success(f'Config loaded successfully!')
        except Exception as e:
            logger.error(f'Error with loading config: {e}')

    def save_config(self):
        self.config_path, _ = QFileDialog.getSaveFileName(filter='*.yaml')
        try:
            connections_config = []
            for conn in self.connections:
                temp_dict = {'name': conn.name,
                             'weights_path': conn.weights_path,
                             'ip': conn.ip,
                             'port': conn.port}
                connections_config.append(temp_dict)
            with open(self.config_path, 'w') as f:
                yaml.dump({'connections': connections_config}, f, sort_keys=False)
            logger.success(f'Config saved successfully!')
        except Exception as e:
            logger.error(f'Error with saving config: {e}')

    def closeEvent(self, event):
        for con in self.connections:
            con.kill_client()

    def handle_visibility_change(self, visible):
        if not visible:
            self.dock_widget.setFloating(False)
            self.dock_widget.setVisible(True)
            self.addDockWidget(Qt.Qt.RightDockWidgetArea, self.dock_widget)

    def start_all(self):
        for connection in self.connections:
            connection.init_client()


class AddingConnectionWindow(QDialog):
    signal_new_connection = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('New connection')
        self.setFixedSize(QSize(400, 170))
        self.create_ui()
        self.weights_path = None
        self.btn_add_connection.clicked.connect(self.add_connection)
        self.btn_open_weights.clicked.connect(self.open_weights_file)

    def create_ui(self):
        self.l_name = QLabel('Name')
        self.le_name = QLineEdit()
        self.le_name.setFixedWidth(180)
        self.l_weights = QLabel('Weights')
        self.btn_open_weights = QPushButton('Open file')
        self.btn_open_weights.setFixedWidth(180)
        self.l_ip_address = QLabel('IP address')
        self.le_ip_address = QLineEdit()
        self.le_ip_address.setText('127.0.0.1')
        self.le_ip_address.setFixedWidth(180)
        self.l_port_numb = QLabel('Port number')
        # self.le_port_numb = QLineEdit()
        # self.le_port_numb.setText('10091')
        # self.le_port_numb.setFixedWidth(180)
        self.spb_port_numb = QSpinBox()
        self.spb_port_numb.setMaximum(66666)
        self.spb_port_numb.setValue(10091)
        self.spb_port_numb.setFixedWidth(180)
        self.spb_port_numb.setSingleStep(1)

        self.btn_add_connection = QPushButton('Add connection')
        self.btn_add_connection.setDisabled(True)

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        name_layout = QVBoxLayout()
        name_layout.addWidget(self.l_name)
        name_layout.addWidget(self.le_name)
        name_layout.addStretch(0)

        weights_layout = QVBoxLayout()
        weights_layout.addWidget(self.l_weights)
        weights_layout.addWidget(self.btn_open_weights)
        weights_layout.addStretch(0)

        header_layout = QHBoxLayout()
        header_layout.addLayout(name_layout)
        header_layout.addLayout(weights_layout)

        ip_layout = QVBoxLayout()
        ip_layout.addWidget(self.l_ip_address)
        ip_layout.addWidget(self.le_ip_address)
        ip_layout.addStretch(0)

        port_layout = QVBoxLayout()
        port_layout.addWidget(self.l_port_numb)
        port_layout.addWidget(self.spb_port_numb)
        port_layout.addStretch(0)

        cntrls_layout = QHBoxLayout()
        cntrls_layout.addLayout(ip_layout)
        cntrls_layout.addLayout(port_layout)

        self.main_layout.addLayout(header_layout)
        self.main_layout.addLayout(cntrls_layout)
        self.main_layout.addSpacing(10)
        self.main_layout.addWidget(self.btn_add_connection)

    def add_connection(self):
        self.signal_new_connection.emit({'name': self.le_name.text(),
                                         'weights_path': self.weights_path,
                                         'ip': self.le_ip_address.text(),
                                         'port': int(self.spb_port_numb.text())})
        self.close()

    def open_weights_file(self):
        self.weights_path, _ = QFileDialog.getOpenFileName(filter='*.pt')
        if self.weights_path[-3:] == '.pt':
            self.btn_add_connection.setEnabled(True)
            logger.success(f'Weights path is {self.weights_path}')
        else:
            logger.error(f'Unknowm file {self.weights_path}')


class DataThread(QThread):
    signal_image = pyqtSignal(np.ndarray)
    signal_recogn_values = pyqtSignal(np.ndarray)
    signal_images_for_save = pyqtSignal(np.ndarray, np.ndarray, dict)
    signal_recogn_df = pyqtSignal(str, dict)

    def __init__(self, q, window_name: str):
        QThread.__init__(self)
        self.q = q
        self.window_name = window_name
        self.start()
        self.save_status = False
        self.trigger_status = False

    def processing_results(self, df: pd.DataFrame):
        self.get_object_freq(df)

        group_res = df.groupby(['name'])['confidence'].max()
        result_dict = {}

        # values = [group_res.get(label) for label in nn.all_classes]

        for label in nn.all_classes:
            result_dict[label] = group_res.get(label)

        for key, value in result_dict.items():
            if value is None:
                result_dict[key] = 0

        return result_dict

    def get_object_freq(self, df: pd.DataFrame):
        max_confidence_rows = df.loc[df.groupby('class')['confidence'].idxmax()]

        # Создание объектов для каждого класса
        objects = []
        for _, row in max_confidence_rows.iterrows():
            detection = RecognitionObject(
                ymin=row['ymin'],
                ymax=row['ymax'],
                confidence=row['confidence'],
                class_id=row['class'],
                name=row['name']
            )
            objects.append(detection)
            print(detection)


    def run(self):
        while True:
            recogn_res = self.q.get()

            self.signal_image.emit(recogn_res['img_res'])
            self.signal_recogn_values.emit(recogn_res['predict_res'])

            processed_dict = self.processing_results(df=recogn_res['predict_df'])

            if self.save_status:
                self.signal_images_for_save.emit(recogn_res['clear_image'],recogn_res['img_res'], processed_dict)

            self.signal_recogn_df.emit(self.window_name, processed_dict)


class RecognitionViewerWidget(QDockWidget, QWidget):
    def __init__(self, ip: str, port: int, window_name: str, weights_path: str):
        super().__init__('Recognition')

        self.name = window_name + f' ({str(port)})'
        self.ip = ip
        self.port = port
        self.weights_path = weights_path
        self.classes = nn.map_list
        self.create_ui()
        self.q = Queue()
        self.process = None
        self.save_path = r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\saved_images"

        self.setWindowTitle(f"{self.name}")

        self.dataThread = DataThread(self.q, self.name)
        self.dataThread.signal_image.connect(self.update_image)
        self.dataThread.signal_recogn_values.connect(self.chart_widget.set_data)

        self.dataThread.signal_images_for_save.connect(self.check_trigger)

        self.last_time = time.time()
        self.fps_deque = deque(maxlen=10)

        self.btn_start.clicked.connect(self.start_button_pressed)
        self.btn_save.clicked.connect(self.change_save_status)

    def create_ui(self):
        self.pixmap = QPixmap()
        self.label_fps = QLabel()
        self.label_fps.setFixedHeight(14)
        self.image_frame = QLabel()
        self.chart_widget = Plot(150, nn.map_list)
        self.btn_start = QPushButton('Start')
        self.btn_start.setFixedWidth(60)
        self.btn_start.setCheckable(True)
        self.btn_save = QPushButton('Save images')
        self.btn_save.setFixedWidth(100)
        self.btn_save.setCheckable(True)
        self.l_cb_trigger_class = QLabel('Class for trigger')
        self.cb_trigger_class = QComboBox()
        self.cb_trigger_class.setFixedWidth(120)
        for name in nn.all_classes:
            self.cb_trigger_class.addItem(name)
        self.cb_trigger_class.addItem('Any')

        self.main_layout = QVBoxLayout()

        start_btn_layout = QVBoxLayout()
        start_btn_layout.addWidget(QLabel())
        start_btn_layout.addWidget(self.btn_start)
        save_img_layout = QVBoxLayout()
        save_img_layout.addWidget(QLabel())
        save_img_layout.addWidget(self.btn_save)
        cb_trigger_layout = QVBoxLayout()
        cb_trigger_layout.addWidget(self.l_cb_trigger_class)
        cb_trigger_layout.addWidget(self.cb_trigger_class)

        header_layout = QHBoxLayout()
        header_layout.addWidget(self.label_fps)
        header_layout.addLayout(start_btn_layout)
        header_layout.addLayout(save_img_layout)
        header_layout.addLayout(cb_trigger_layout)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.image_frame)
        chart_layout = QHBoxLayout()
        chart_layout.addWidget(self.chart_widget)

        splitter = QSplitter(Qt.Qt.Vertical)
        splitter.addWidget(self.image_frame)
        splitter.addWidget(self.chart_widget)
        splitter.setSizes([50, 500])  # Initial sizes

        self.main_layout.addLayout(header_layout)
        self.main_layout.addLayout(image_layout)
        self.main_layout.addLayout(chart_layout)

        main_widget = QWidget()
        main_widget.setLayout(self.main_layout)
        self.setWidget(main_widget)

        self.disply_width = 640
        self.display_height = 640

        # create the label that holds the image
        self.image_frame.setMinimumSize(640, 640)  # Allow resizing to smaller sizes
        self.image_frame.setSizePolicy(Qt.QSizePolicy.Expanding, Qt.QSizePolicy.Expanding)
        self.chart_widget.setMinimumSize(100, 180)  # Allow resizing to smaller sizes
        self.image_frame.setSizePolicy(Qt.QSizePolicy.Expanding, Qt.QSizePolicy.Expanding)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_frame with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_frame.setPixmap(qt_img)
        time_now = time.time()
        self.fps_deque.appendleft(1 / (time_now - self.last_time))
        self.label_fps.setText(f'FPS = {(sum(self.fps_deque) / len(self.fps_deque)):.2f}')
        self.last_time = time_now

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_frame.width(), self.image_frame.height(), Qt.Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def show_image(self, image):
        h, w, ch = image.shape
        image = QImage(image.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
        self.image_frame.setPixmap(QPixmap.fromImage(image))
        time_now = time.time()
        self.label_fps.setText(f'FPS = {1 / (time_now - self.last_time)}')
        self.last_time = time_now

    def init_client(self):
        if self.process is None:
            self.process = nn.Client((self.ip, self.port), weights_path=self.weights_path)
            self.process.set_queue(self.q)
            self.process.start()
            self.btn_start.setText('Stop')
            self.btn_start.setChecked(True)

    def kill_client(self):
        if self.process is not None:
            self.process.kill()
            self.process = None
            self.btn_start.setText('Start')
            self.btn_start.setChecked(False)

    def start_button_pressed(self):
        try:
            if self.btn_start.isChecked():
                self.init_client()

            if not self.btn_start.isChecked():
                self.kill_client()
                logger.info(f'Process \'{self.name}\' was killed.')
        except Exception as e:
            logger.error(e)

    def change_save_status(self):
        try:
            if self.btn_save.isChecked():
                self.btn_save.setText('Stop save')
                self.dataThread.save_status = True
                logger.info('Saving images...')

            if not self.btn_save.isChecked():
                self.btn_save.setText('Save images')
                self.dataThread.save_status = False
                logger.info('Saving stopped.')
        except Exception as e:
            logger.error(e)

    def check_trigger(self, clear_arr, detected_arr, result_dict):
        try:
            if self.cb_trigger_class.currentText() == 'Any':
                self.save_image(clear_arr, detected_arr)
            elif result_dict[self.cb_trigger_class.currentText()] > 0.2:
                self.save_image(clear_arr, detected_arr)
        except KeyError as e:
            logger.error(e)

    def save_image(self, clear_arr, detected_arr):
        color_clear_image = cv2.applyColorMap(clear_arr, cv2.COLORMAP_RAINBOW)
        screen = cv2.resize(color_clear_image, (640, 640))
        filename = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        if not cv2.imwrite(filename=self.save_path + '\\' + filename + '.jpg', img=screen):
            logger.error('Error with saving images!')

        color_detected_image = cv2.applyColorMap(detected_arr, cv2.COLORMAP_RAINBOW)
        screen = cv2.resize(color_detected_image, (640, 640))
        if not cv2.imwrite(filename=self.save_path + '\\' + filename + '_detected.jpg', img=screen):
            logger.error('Error with saving images!')


class Plot(pyqtgraph.PlotWidget):
    def __init__(self, max_history_size, classes):
        super().__init__()
        self.classes = classes
        self.colors = []
        for name in self.classes:
            self.colors.append(GLOBAL_COLORS[name])
        self.conf = {}
        self.max_history_size = max_history_size
        self.counter = 0
        self.history_size = 0
        self.buffer = np.zeros(shape=(len(self.classes), max_history_size), dtype=np.float64)
        self.curves = None
        self.create_curves()

    def create_curves(self):
        self.curves = []

        legend = pyqtgraph.LegendItem((20, 0), offset=(25, 1))
        legend.setParentItem(self.getPlotItem())
        legend.setBrush(pyqtgraph.mkBrush((50, 50, 50, 150)))  # Прозрачность фона

        for i in range(len(self.classes)):
            curve = self.plot(name=str(i), pen=self.colors[i])
            curve.setPen(self.colors[i])
            self.curves.append(curve)
            legend.addItem(curve, self.classes[i])

        for sample, label in legend.items:
            # Установить стиль шрифта
            font = pyqtgraph.QtGui.QFont()
            font.setPointSize(5)  # Размер шрифта
            label.setFont(font)
            label.setAttr('color', 'white')  # Цвет шрифта

    def append(self, data):
        self.counter += 1
        if self.history_size < self.max_history_size:
            self.history_size += 1
        for i in range(len(data)):
            self.buffer[i] = np.roll(self.buffer[i], -1, axis=0)
            self.buffer[i][-1] = data[i]

    def set_data(self, data):
        self.append(data)
        if self.history_size < self.max_history_size:
            data_view = self.buffer[:, -self.history_size:]
        else:
            data_view = self.buffer
        for i in range(len(self.curves)):
            self.curves[i].setData(y=data_view[i])


class HistogramViewerWidget(QDockWidget, QWidget):

    def __init__(self, window_name: str):
        super().__init__('Histograms')
        self.classes = nn.all_classes
        self.numb_of_classes = (len(self.classes))
        self.docks = {}

        self.area = DockArea()
        self.setWidget(self.area)
        self.create_histogram(window_name)

    def create_histogram(self, window_name: str):
        # w1 = pyqtgraph.LayoutWidget()
        dock = DockGraph(window_name, self.classes)
        self.docks[window_name] = dock
        self.area.addDock(dock, 'bottom')

    def update_data(self, window_name, result_dict: dict):
        self.docks[window_name].update_plot(list(result_dict.values()))


class DockGraph(Dock):
    def __init__(self, window_name: str, classes):
        super().__init__(window_name)

        self.accum_size = 50
        self.classes = classes
        self.colors = []
        for name in self.classes:
            self.colors.append(GLOBAL_COLORS[name])
        self.plot = pyqtgraph.plot()
        self.plot.setYRange(0, self.accum_size, 0)
        self.plot.showAxis('left', True)

        # Set names om X axis
        ticks = []
        for i in range(len(self.classes)):
            ticks.append((i, self.classes[i]))
        ticks = [ticks]
        ax = self.plot.getAxis('bottom')
        ax.setTicks(ticks)
        self.addWidget(self.plot, row=0, colspan=len(self.classes))
        self.indicators = {}
        i = 0
        for name in nn.all_classes:
            self.indicators[name] = QPushButton(name)
            self.indicators[name].setCheckable(True)

            self.indicators[name].setStyleSheet("QPushButton:open {"
                                                "background-color: rgb" + str(GLOBAL_COLORS[name]) + ";"
                                                "}")
            self.addWidget(self.indicators[name], row=1, col=i)
            i += 1

        self.deques = [deque(maxlen=self.accum_size) for i in range(len(self.classes))]

    def make_decision(self, accumed_val):
        for i in range(len(self.indicators)):
            if accumed_val[i] >= 12.5:
                state = True
            else:
                state = False
            list(self.indicators.values())[i].setChecked(state)

    def update_plot(self, values):
        for i in range(len(values)):
            self.deques[i].appendleft(values[i])

        accumed_val = [sum(deq) for deq in self.deques]
        self.make_decision(accumed_val)

        # Clear plot
        for item in self.plot.items():
            self.plot.removeItem(item)

        # Draw bars
        bar_item = pyqtgraph.BarGraphItem(x=np.arange(len(self.classes)), height=accumed_val,
                                          width=0.7, brushes=self.colors)
        self.plot.addItem(bar_item)


class RecognitionObject:
    def __init__(self, ymin, ymax, confidence, class_id, name):
        super().__init__()
        self.center_freq, self.width_freq = self.calculate_frequency(ymin, ymax)
        self.confidence = confidence
        self.class_id = class_id
        self.name = name

    def calculate_frequency(self, ymin, ymax):
        """ Функция считает частоту сигнала относительно центральной частоты (нулевой) """

        f_min = (nn.FREQ_SMPLS / nn.IMG_SIZE[0]) * ymin
        f_max = (nn.FREQ_SMPLS / nn.IMG_SIZE[0]) * ymax
        width = f_max - f_min
        f = width / 2 + f_min
        f_center = nn.FREQ_SMPLS / 2 - f
        return f_center * (-1), width

    def __repr__(self):
        return (f"Detection( name={self.name}, "
                f"class_id={self.class_id}, "
                f"confidence={self.confidence}, "
                f"center_frequency={self.center_freq}, "
                f"width_frequency={self.width_freq})")


def main_gui():
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':

    PORTS = [10092]
    HOST = "127.0.0.1"

    q = Queue()
    gui_process = Process(target=main_gui)
    gui_process.start()


