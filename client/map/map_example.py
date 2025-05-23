import sys
from offline_folium import offline
import folium
import os
from PySide6.QtWidgets import QApplication, QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy
import PySide6.QtWebEngineWidgets as QtWebEngineWidgets
from PySide6.QtCore import QUrl, Qt
from collections import namedtuple
import io


UAVObject = namedtuple('UAV',
                       ['id', 'name', 'position_history', 'altitude', 'pilot_position', 'serial_number'])

# Отключение GPU (если требуется для вашей системы)
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --disable-web-security"


class MapWidget(QDockWidget, QWidget):
    def __init__(self, map_settings: dict):
        super().__init__()
        self.map_settings = map_settings
        self.setWindowTitle("Map")
        self.resize(800, 600)

        self.setMinimumSize(300, 300)

        tiles_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../map", "tiles"))
        tiles_path = f"file:///{tiles_folder}/{{z}}/{{x}}/{{y}}.png".replace('\\', '/')

        # create Folium map with offline tiles
        self.map = folium.Map(
            location=self.map_settings['base_position'],  # Координаты центра
            zoom_start=self.map_settings['zoom_start'],
            tiles=tiles_path,  # Путь к локальным тайлам
            attr="Offline Tiles (© OpenStreetMap contributors)",
            min_zoom=self.map_settings['min_zoom'],
            max_zoom=self.map_settings['max_zoom'],
            crs="EPSG3857",
            control_scale=True,
            no_touch=False  # Отключаем интерактивность, если не нужна
        )

        # Добавляем маркер
        folium.Marker(location=self.map_settings['base_position'],
                      popup="You",
                      tooltip="Your position",
                      icon=folium.CustomIcon(icon_image=f'../assets/icons/map/antenna.png',
                                             icon_size=(28, 28),  # Adjust based on your icon dimensions
                                             icon_anchor=(16, 28),  # Adjust to center the icon
                                             popup_anchor=(0, -28))).add_to(self.map)
        folium.Circle(location=self.map_settings['base_position'],
                      radius=1000,
                      fill_color='blue',
                      stroke=False).add_to(self.map)
        folium.Circle(location=self.map_settings['base_position'], radius=500, fill_color='red', stroke=False).add_to(self.map)
        points = [[53.9325706, 27.6451251],
                  [53.9332896, 27.6479315],
                  [53.9308304, 27.6468779]]

        # Сохраняем карту в HTML
        html_file = os.path.abspath("map.html")
        self.map.save(html_file)

        # Создаем веб-просмотрщик
        self.web_view = QtWebEngineWidgets.QWebEngineView()
        self.setWidget(self.web_view)

        # Настраиваем разрешения для локальных файлов
        settings = self.web_view.page().settings()
        settings.setAttribute(settings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(settings.WebAttribute.LocalContentCanAccessRemoteUrls, True)

        # self.main_layout.addWidget(self.web_view)

        # Загружаем HTML-файл
        self.web_view.load(QUrl.fromLocalFile(html_file))

        self.add_object(uav_object=UAVObject(id='asalam',
                                             name='DJI Phantom',
                                             position_history=[
                                                                [53.9325706, 27.6451251],
                                                                [53.9332896, 27.6479315],
                                                                [53.9308304, 27.6468779],
                                                                [53.9305065, 27.6437918],
                                                                [53.9311214, 27.6398012]],
                                             altitude=127,
                                             pilot_position=[53.9308304, 27.6468779],
                                             serial_number='534645yfgsd5'))

    def add_object(self, uav_object: UAVObject):
        print('ADD new obj to map')
        folium.Marker(location=uav_object.position_history[-1],
                      popup=f'{uav_object.name} H={uav_object.altitude}m',
                      tooltip=uav_object.serial_number,
                      icon=folium.CustomIcon(icon_image=f'../assets/icons/map/drone_black.png',
                                             icon_size=(28, 28),  # Adjust based on your icon dimensions
                                             icon_anchor=(16, 28),  # Adjust to center the icon
                                             popup_anchor=(0, -28))).add_to(self.map)

        folium.Marker(location=uav_object.pilot_position,
                      tooltip='Pilot',
                      icon=folium.CustomIcon(icon_image=f'../assets/icons/map/remote-control.png',
                                             icon_size=(28, 28),  # Adjust based on your icon dimensions
                                             icon_anchor=(16, 28),  # Adjust to center the icon
                                             popup_anchor=(0, -28))).add_to(self.map)

        polyline = folium.PolyLine(locations=uav_object.position_history,
                                   color='green',
                                   dash_array='7, 10').add_to(self.map)
        self.update_map()

    def update_map(self):
        """Обновляет HTML карты"""
        html_file = os.path.abspath("map.html")
        self.map.save(html_file)
        self.web_view.load(QUrl.fromLocalFile(html_file))


def main():
    app = QApplication(sys.argv)
    window = MapWidget(map_settings={'base_position': [53.9312229, 27.6358432],
  'zoom_start': 15,
  'min_zoom': 10,
  'max_zoom': 17})
    window.show()
    sys.exit(app.exec())




if __name__ == "__main__":
    main()