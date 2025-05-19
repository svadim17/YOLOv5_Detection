import sys
import folium
import os
import PySide6.QtWidgets as QtWidgets
import PySide6.QtWebEngineWidgets as QtWebEngineWidgets
from PySide6.QtCore import QUrl

# Отключение GPU (если требуется для вашей системы)
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"


class MapWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Offline Folium Map with PySide6")
        self.resize(800, 600)

        # Путь к локальным тайлам
        tiles_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "map", "../../map/tiles"))
        tiles_path = f"file:///{tiles_folder}/{{z}}/{{x}}/{{y}}.png".replace('\\', '/')
        print(f'tiles_path = {tiles_path}')

        # Создаем карту Folium с оффлайн-тайлами
        m = folium.Map(
            location=[53.90366, 27.56938],  # Координаты центра
            zoom_start=10,
            tiles=tiles_path,  # Путь к локальным тайлам
            attr="Offline Tiles (© OpenStreetMap contributors)",
            min_zoom=10,
            max_zoom=17,
            crs="EPSG3857",
            control_scale=True,
            no_touch=True  # Отключаем интерактивность, если не нужна
        )

        # Добавляем маркер
        folium.Marker(
            [53.90366, 27.56938],
            popup="Park",
            tooltip="Click me!"
        ).add_to(m)

        # Сохраняем карту в HTML
        html_file = os.path.abspath("map.html")
        m.save(html_file)

        # Создаем веб-просмотрщик
        self.web_view = QtWebEngineWidgets.QWebEngineView()

        # Настраиваем разрешения для локальных файлов
        settings = self.web_view.page().settings()
        settings.setAttribute(settings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(settings.WebAttribute.LocalContentCanAccessRemoteUrls, True)

        self.setCentralWidget(self.web_view)

        # Загружаем HTML-файл
        self.web_view.load(QUrl.fromLocalFile(html_file))


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MapWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()