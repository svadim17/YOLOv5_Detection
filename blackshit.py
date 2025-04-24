class CircularProgress(QWidget):
    def __init__(self, label, color=QColor("#00ffff")):
        super().__init__()
        self.value = 0
        self.label = label
        self.color = color
        self.setMinimumSize(50, 50)

    def setValue(self, val):
        self.value = val
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        pen_width = 10
        radius = int(min(rect.width(), rect.height()) / 2 - pen_width)

        center = rect.center()
        painter.translate(center)
        painter.rotate(-90)

        arc_rect = QRect(-radius, -radius, radius * 2, radius * 2)

        # Draw background circle
        pen = QPen(QColor("#222"), pen_width)
        painter.setPen(pen)
        painter.drawArc(arc_rect, 0, 360 * 16)

        # Draw value arc
        pen.setColor(self.color)
        painter.setPen(pen)
        angle = int(360 * self.value / 100)
        painter.drawArc(arc_rect, 0, -angle * 16)

        # Draw text
        painter.resetTransform()
        painter.setPen(QColor("white"))
        font = QFont("Arial", 16, QFont.Weight.Bold)
        painter.setFont(font)
        text = f"{self.label}\n{self.value:.0f}%"
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)