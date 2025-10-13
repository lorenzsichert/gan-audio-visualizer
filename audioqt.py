import sys
import os
import random
import numpy as np
import torch
import sounddevice as sd
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QMainWindow,
    QAction, QDialog, QSlider, QFormLayout, QDialogButtonBox,
    QSizePolicy, QLineEdit, QPushButton, QSpinBox, QFileDialog,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter

from dcgan import ColorGenerator
from recording import get_sample

from options_dialog import OptionsDialog
from models_dialog import ModelsDialog


class GANVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audio GAN Visualizer (Qt)")
        self.width, self.height = 250, 250
        self.resize(self.width, self.height)

        # --- Central Widget ---
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # --- Set Minimum Size ---
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setMinimumSize(50, 50) 

        # --- Audio and Model Setup ---
        self.rate = 44100
        self.blocksize = 1000
        self.device = 'default'
        self.latent_dim = 100
        self.image_size = 256
        self.image_channels = 1
        self.model_path = "models/256,256,1/andylomas100white.pth"

        self.generator = ColorGenerator(self.image_size, self.latent_dim, self.image_channels)
        self.generator.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.generator.eval()

        # --- Tiling Settings ---
        self.tiling = 2
        self.current_pixmap = QPixmap(self.width, self.height)
        self.tile_pixmap = QPixmap(self.width // self.tiling,
                                   self.height // self.tiling)

        # Parameters
        self.smoothing_factor = 0.6
        self.noise_weight = 0.3
        self.audio_weight = 0.3


        self.smoothed_spectrum = np.zeros(int(self.blocksize / 2) + 1)
        self.lookup = np.arange(self.latent_dim)
        self.a = torch.randn(1, self.latent_dim)
        self.b = torch.randn(1, self.latent_dim)
        self.step = 0

        # --- Start Audio Stream ---
        self.stream = sd.InputStream(
            samplerate=self.rate,
            blocksize=self.blocksize,
            channels=1,
            dtype='float32',
            device=self.device
        )
        self.stream.start()

        # --- Timer for updates ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # --- Menu ---
        self.init_menu()

    def init_menu(self):
        menubar = self.menuBar()
        options_menu = menubar.addMenu("Options")

        adjust_action = QAction("Adjust Sliders", self)
        adjust_action.triggered.connect(self.open_options_dialog)
        options_menu.addAction(adjust_action)

        load_action = QAction("Load Models", self)
        load_action.triggered.connect(self.open_models_dialog)
        options_menu.addAction(load_action)

    def open_options_dialog(self):
        dialog = OptionsDialog(self)
        dialog.exec_()

    def open_models_dialog(self):
        dialog = ModelsDialog(self)
        dialog.exec_()

    def update_frame(self):
        self.step += 1
        self.smoothed_spectrum = get_sample(
            self.stream, self.smoothed_spectrum, self.blocksize, self.smoothing_factor
        )

        if self.step % 10 == 0:
            c = random.randint(0, self.latent_dim - 1)
            d = random.randint(0, self.latent_dim - 1)
            self.lookup[d], self.lookup[c] = self.lookup[c], self.lookup[d]

        if self.step % 20 == 0:
            self.a[0][random.randint(0, self.latent_dim - 1)] = torch.randn(1)[0]
            self.b[0][random.randint(0, self.latent_dim - 1)] = torch.randn(1)[0]

        spectrum = np.zeros(self.latent_dim)
        for i in range(self.latent_dim):
            spectrum[i] = self.smoothed_spectrum[self.lookup[i]]

        noise = torch.zeros(1, self.latent_dim)
        for i in range(self.latent_dim):
            noise[0][i] = self.a[0][i] * self.noise_weight + spectrum[i] * self.audio_weight * self.b[0][i]

        image = self.generator(noise).detach().squeeze()
        image_array = ((image.numpy() + 1) / 2.0 * 255).astype(np.uint8)
        if self.image_channels == 1:
            image_array = np.stack([image_array] * 3, axis=0)
        image_rgb = np.transpose(image_array, (1, 2, 0))
        image_rgb = np.ascontiguousarray(image_rgb)

        qimage = QImage(
            image_rgb.data,
            image_rgb.shape[1],
            image_rgb.shape[0],
            image_rgb.strides[0],
            QImage.Format_RGB888
        )

        self.tile_pixmap = QPixmap.fromImage(qimage).scaled(
            int(self.width / self.tiling)+1,
            int(self.height / self.tiling)+1,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        painter = QPainter(self.current_pixmap)

        for x in range(0, self.width, int(self.width / self.tiling)+1):
            for y in range(0, self.height, int(self.height / self.tiling)+1):
                painter.drawPixmap(x, y, self.tile_pixmap)
        painter.end()

        self.update_scaled_pixmap()

    def update_scaled_pixmap(self):
        """Rescale the pixmap to match window size while keeping aspect ratio."""
        if self.current_pixmap:
            scaled_pixmap = self.current_pixmap.scaled(
                self.label.size(),
                Qt.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.label.setPixmap(scaled_pixmap)

    def reload_generator(self):
        """Recreate the generator and load weights."""
        try:
            # Recreate the generator (ensures architecture matches)
            self.generator = ColorGenerator(self.image_size, self.latent_dim, self.image_channels)
            state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
            self.generator.load_state_dict(state_dict)
            self.generator.eval()
            print(f"✅ Reloaded generator from {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to reload model: {e}")

    def resizeEvent(self, event):
        """Automatically rescale image when window is resized."""
        super().resizeEvent(event)
        self.update_scaled_pixmap()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        self.stream.stop()
        self.stream.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    visualizer = GANVisualizer()
    visualizer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
