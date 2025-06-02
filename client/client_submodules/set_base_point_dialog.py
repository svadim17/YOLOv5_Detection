from PyQt6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QDialogButtonBox


class CoordinateInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Base Point Coordinates")
        self.setMinimumWidth(300)

        # Поля ввода
        self.lat_input = QLineEdit()
        self.lng_input = QLineEdit()

        form_layout = QFormLayout()
        form_layout.addRow("Latitude:", self.lat_input)
        form_layout.addRow("Longitude:", self.lng_input)

        # Кнопки OK / Cancel
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def get_coordinates(self):
        try:
            lat = float(self.lat_input.text())
            lng = float(self.lng_input.text())
            return lat, lng
        except ValueError:
            return None, None
