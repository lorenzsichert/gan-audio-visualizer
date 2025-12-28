from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt

class FullscreenImageWindow(QWidget):
    def __init__(self, pixmap):
        super().__init__()
        self.setWindowTitle("Image Viewer")
        self.setStyleSheet("background-color: black;")
        
        # Fullscreen
        self.showFullScreen()

        # QLabel to display the image
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.update_pixmap(pixmap)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(0)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        # Resize the image when the window is resized
        self.resizeEvent = self.on_resize

    def on_resize(self, _): 
        if not self.pixmap.isNull():
            self.label.setPixmap(self.pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

    def update_pixmap(self, pixmap):
        self.pixmap = pixmap
        self.label.setPixmap(self.pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        if self.stream != None:
            self.stream.stop()
            self.stream.close()
        event.accept()
