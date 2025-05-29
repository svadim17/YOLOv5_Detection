import sys
import folium
import os
from PySide6.QtWidgets import QApplication, QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy
import PySide6.QtWebEngineWidgets as QtWebEngineWidgets
from PySide6.QtCore import QUrl, Qt, QTimer
from collections import namedtuple
from loguru import logger
import json



UAVObject = namedtuple('UAV',
                       ['id', 'name', 'position_history', 'altitude', 'pilot_position', 'serial_number'])

# Отключение GPU (если требуется для вашей системы)
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --disable-web-security"


class MapWidget(QDockWidget, QWidget):
    def __init__(self, map_settings: dict):
        super().__init__()
        self.map_settings = map_settings
        self.setTitleBarWidget(QWidget())
        # self.setWindowTitle("Map")
        # self.resize(800, 600)
        self.setMinimumSize(300, 300)

        tiles_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../map", "tiles"))
        self.tiles_path = f"file:///{tiles_folder}/{{z}}/{{x}}/{{y}}.png".replace('\\', '/')
        self.leaflet_css = f"file:///{os.path.abspath('assets/leaflet/leaflet.css')}".replace('\\', '/')
        self.leaflet_js = f"file:///{os.path.abspath('assets/leaflet/leaflet.js')}".replace('\\', '/')

        # Создаем веб-просмотрщик
        self.web_view = QtWebEngineWidgets.QWebEngineView()
        self.setWidget(self.web_view)

        # Настраиваем разрешения для локальных файлов
        settings = self.web_view.page().settings()
        settings.setAttribute(settings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(settings.WebAttribute.LocalContentCanAccessRemoteUrls, True)

        # Загружаем базовый HTML с картой
        self.load_base_map()

        # Тестовые данные
        self.markers = {}
        self.trajectories = {}
        self.trajectory_points = []  # Для хранения точек пути

        # Ждем загрузки страницы перед добавлением маркера
        self.web_view.loadFinished.connect(self.on_load_finished)

    def load_base_map(self):
        """Загружает базовую карту OpenStreetMap"""
        # Абсолютные пути к файлам
        logger.debug(f"Tiles path: {self.tiles_path}")
        logger.debug(f"Leaflet CSS: {self.leaflet_css}")
        logger.debug(f"Leaflet JS: {self.leaflet_js}")

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>OSM Map</title>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" href="{self.leaflet_css}" />
            <script src="{self.leaflet_js}"></script>
            <style>
                body {{ margin: 0; padding: 0; }}
                #map {{ height: 100vh; width: 100%; }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                console.log('Initializing map...');
                var map = L.map('map').setView([53.9312229, 27.6358432], 15);
                L.tileLayer('{self.tiles_path}', {{
                    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                    minZoom: 10,
                    maxZoom: 17
                }}).addTo(map);

                // Глобальные переменные для хранения объектов
                window.markers = {{}};
                window.trajectories = {{}};

                // Функция для добавления маркера
                function addMarker(id, lat, lng, popup, icon_path, altitude) {{
                    console.log('Adding marker:', id, lat, lng, popup, altitude, icon_path);
                    var markerOptions = {{}};
                    markerOptions.icon = L.icon({{
                            iconUrl: icon_path,
                            iconSize: [28, 28],
                            iconAnchor: [14, 28],
                            popupAnchor: [0, -28]
                        }});
                    var marker = L.marker([lat, lng], markerOptions).addTo(map);
                    if (popup) {{
                        marker.bindPopup(popup + (altitude ? ' (H=' + altitude + 'm)' : ''));
                    }}
                    window.markers[id] = marker;
                    return marker;
                }}

                // Функция для перемещения маркера
                function moveMarker(id, lat, lng, altitude) {{
                    console.log('Moving marker:', id, lat, lng, altitude);
                    if (window.markers[id]) {{
                        window.markers[id].setLatLng([lat, lng]);
                        if (altitude !== undefined) {{
                            window.markers[id].setPopupContent(
                                window.markers[id].getPopup().getContent().split('(H=')[0] + 
                                '(H=' + altitude + 'm)'
                            );
                        }}
                    }}
                }}

                // Функция для добавления пути
                function addTrajectory(id, points, color) {{
                    console.log('Adding trajectory:', id, points, color);
                    var trajectory = L.polyline(points, {{color: color, dashArray: '7, 10'}}).addTo(map);
                    window.trajectories[id] = trajectory;
                    return trajectory;
                }}

                // Функция для обновления пути
                function updateTrajectory(id, points) {{
                    console.log('Updating trajectory:', id, points);
                    if (window.trajectories[id]) {{
                        window.trajectories[id].setLatLngs(points);
                    }}
                }}
                
                // Удаление маркера
                function removeMarker(id) {{
                    console.log('Removing marker:', id);
                    if (window.markers[id]) {{
                        map.removeLayer(window.markers[id]);
                        delete window.markers[id];
                    }}
                }}
                
                // Удаление пути (Polyline)
                function removeTrajectory(id) {{
                    console.log('Removing path:', id);
                    if (window.trajectories[id]) {{
                        map.removeLayer(window.trajectories[id]);
                        delete window.trajectories[id];
                    }}
                }}
            </script>
        </body>
        </html>
        """

        # Сохраняем во временный файл и загружаем
        temp_file = os.path.abspath("temp_map.html")
        logger.debug(f"Saving HTML to: {temp_file}")
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(html)

        self.web_view.setUrl(QUrl.fromLocalFile(temp_file))

    def on_load_finished(self, status):
        """Вызывается после загрузки страницы"""
        if status:
            logger.debug("Page loaded successfully")
            icon_path = f"file:///{os.path.abspath('assets/icons/map/drone_black.png')}".replace('\\', '/')
            # Добавляем тестовый маркер после загрузки страницы
            self.add_marker(marker_id="drone1",
                            lat=53.9312229,
                            lng=27.6358432,
                            name="Test Drone",
                            icon_path=icon_path,
                            altitude=100)
            # Добавляем путь
            self.add_trajectory(trajectory_id="drone1", points=[[53.9312229, 27.6358432]], color="green")
        else:
            logger.error("Failed to load page")

    def add_marker(self, marker_id, lat, lng, name, icon_path, altitude=None):
        """Добавляет новый маркер на карту"""
        logger.debug(f"Adding marker: {marker_id} at {lat}, {lng}")
        js_code = f"addMarker('{marker_id}', {lat}, {lng}, '{name}', '{icon_path}', {altitude});"
        self.web_view.page().runJavaScript(js_code)
        self.markers[marker_id] = {'lat': lat, 'lng': lng}

    def add_trajectory(self, trajectory_id, points, color):
        """Добавляет новый путь на карту"""
        logger.debug(f"Adding path: {trajectory_id} with points {points}")
        js_code = f"addTrajectory('{trajectory_id}', {json.dumps(points)}, '{color}');"
        self.web_view.page().runJavaScript(js_code)
        self.trajectories[trajectory_id] = points

    def update_path(self, trajectory_id, points):
        """Обновляет существующий путь"""
        logger.debug(f"Updating path: {trajectory_id} with points {points}")
        js_code = f"updateTrajectory('{trajectory_id}', {json.dumps(points)});"
        self.web_view.page().runJavaScript(js_code)
        self.trajectories[trajectory_id] = points

    def move_marker(self, marker_id, lat, lng, altitude=None):
        """Перемещает существующий маркер"""
        if marker_id in self.markers:
            logger.debug(f"Moving marker: {marker_id} to {lat}, {lng}")
            js_code = f"moveMarker('{marker_id}', {lat}, {lng}, {altitude});"
            self.web_view.page().runJavaScript(js_code)
            self.markers[marker_id] = {'lat': lat, 'lng': lng}
            # Обновляем путь
            self.trajectory_points.append([lat, lng])
            self.update_path(trajectory_id=marker_id, points=self.trajectory_points)

    def map_emulation(self):
        """Настраивает тестовое перемещение маркера"""
        self.trajectory = [
            {"lat": 53.938040, "lng": 27.672249},
            {"lat": 53.937529, "lng": 27.665938},
            {"lat": 53.935161, "lng": 27.658759},
            {"lat": 53.932281, "lng": 27.650555},
            {"lat": 53.936136, "lng": 27.650634},
            {"lat": 53.938597, "lng": 27.658286},
            {"lat": 53.941337, "lng": 27.667516},
            {"lat": 53.942080, "lng": 27.672407},
            {"lat": 53.938226, "lng": 27.673038},
        ]

        self.current_point = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.move_to_next_point)
        self.timer.start(1000)  # Обновление каждую секунду

    def move_to_next_point(self):
        """Перемещает маркер к следующей точке траектории"""
        if self.current_point < len(self.trajectory):
            point = self.trajectory[self.current_point]
            altitude = 100 + self.current_point * 10
            logger.debug(f"Moving to point {self.current_point}: {point}")
            self.move_marker(marker_id="drone1", lat=point["lat"], lng=point["lng"], altitude=altitude)
            self.current_point += 1
        else:
            self.web_view.page().runJavaScript(f"removeMarker('drone1');")
            self.web_view.page().runJavaScript(f"removeTrajectory('drone1');")

            logger.debug("Stopping timer: reached end of trajectory")
            self.timer.stop()


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