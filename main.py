import random
import sys

from PyQt5.QtCore import QSettings, QTimer, Qt
from PyQt5.QtGui import QImage, QPainter, QPixmap, QTransform
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
import numpy as np
import sounddevice as sd
import torch

from dcgan import Generator
from models_dialog import ModelsDialog
from options_dialog import OptionsDialog
from input_dialog import InputDialog
from recording import get_sample


class GANVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audio GAN Visualizer (Qt)")
        self.label_width, self.label_height = 500, 500
        self.resize(self.label_width, self.label_height)

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

        """ Load and Save Settings """
        self.settings = QSettings("lorenzsichert", "GANVisualizer")

        saved_device = self.settings.value("input_device", defaultValue=None)
        if saved_device is not None:
            print("Found saved device.")
            self.device = saved_device
        else:
            self.device = None

        # --- Audio and Model Setup ---
        self.blocksize = 1000
        self.latent_dim = 100
        self.image_size = 64
        self.image_channels = 3
        self.model_path = "models/64,64,3/generator-23.pth"

        self.reload_generator()

        # --- Tiling Settings ---
        self.tiling = 1
        self.current_pixmap = QPixmap(self.label_width, self.label_height)
        self.tile_pixmap = QPixmap(self.label_width // self.tiling,
                                   self.label_height // self.tiling)

        # Parameters
        self.smoothing_factor = 0.6
        self.noise_weight = 0.4
        self.audio_weight = 0.1
        self.noise_randomization = 4
        self.audio_randomization = 2


        self.smoothed_spectrum = np.zeros(int(self.blocksize / 2) + 1)
        self.lookup = np.arange(self.latent_dim)
        self.a = torch.randn(1, self.latent_dim)
        self.b = torch.randn(1, self.latent_dim)
        self.step = 0

        # --- Start Audio Stream ---
        self.stream = None
        self.open_stream()

        # --- Timer for updates ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # --- Menu ---
        self.init_menu()

    def open_stream(self):
        print(f"Opening Stream: {self.device}")
        if self.device == None:
            self.stream = None
            return
        
        if self.device['max_input_channels'] < 1:
            print(f"❌ {self.device['name']} has no input channels.")
            return

        try:
            self.stream = sd.InputStream(
                samplerate=self.device['default_samplerate'],
                blocksize=self.blocksize,
                channels=self.device['max_input_channels'],
                dtype='float32',
                device=self.device['name']
            )
            self.stream.start()
            print("✅ Stream started succesfully!")
        except sd.PortAudioError as e:
            print(f"❌ Failed to start Stream: {e}")
            self.stream = None
        except Exception as e:
            print(f"❌ Failed to start Stream: {e}")
            self.stream.stop()
            self.stream.close()
            self.stream = None


    def init_menu(self):
        menubar = self.menuBar()
        options_menu = menubar.addMenu("Options")

        adjust_action = QAction("Adjust Sliders", self)
        adjust_action.triggered.connect(self.open_options_dialog)
        options_menu.addAction(adjust_action)

        load_action = QAction("Load Models", self)
        load_action.triggered.connect(self.open_models_dialog)
        options_menu.addAction(load_action)

        input_action = QAction("Input Settings", self)
        input_action.triggered.connect(self.open_input_dialog)
        options_menu.addAction(input_action)

    def open_options_dialog(self):
        dialog = OptionsDialog(self)
        dialog.setModal(False)
        dialog.show()

    def open_models_dialog(self):
        dialog = ModelsDialog(self)
        dialog.setModal(False)
        dialog.show()

    def open_input_dialog(self):
        dialog = InputDialog(self)
        dialog.setModal(False)
        dialog.show()

    def update_frame(self):
        self.step += 1
        if self.stream != None:
            self.smoothed_spectrum = get_sample(
                self.stream, self.smoothed_spectrum, self.blocksize, self.smoothing_factor
            )
        else:
            self.smoothed_spectrum = np.zeros(int(self.blocksize / 2) + 1)

        # --- Randomize Latent Vector ---
        if self.audio_randomization != 0 and self.step % int(30 / self.audio_randomization) == 0:
            c = random.randint(0, self.latent_dim - 1)
            d = random.randint(0, self.latent_dim - 1)
            self.lookup[d], self.lookup[c] = self.lookup[c], self.lookup[d]

        if self.noise_randomization != 0 and self.step % int(30 / self.noise_randomization) == 0:
            self.a[0][random.randint(0, self.latent_dim - 1)] = torch.randn(1)[0]
            self.b[0][random.randint(0, self.latent_dim - 1)] = torch.randn(1)[0]

        spectrum = np.zeros(self.latent_dim)
        for i in range(self.latent_dim):
            spectrum[i] = self.smoothed_spectrum[self.lookup[i]]

        noise = torch.zeros(1, self.latent_dim)
        for i in range(self.latent_dim):
            noise[0][i] = self.a[0][i] * self.noise_weight + spectrum[i] * self.audio_weight * self.b[0][i]

        noise = noise.view(1,100,1,1)
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
            int(self.label_width / self.tiling)+1,
            int(self.label_height / self.tiling)+1,
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )

        painter = QPainter(self.current_pixmap)

        i = 0
        j = 0
        for x in range(0, self.label_width, int(self.label_width / self.tiling)+1):
            j += 1
            for y in range(0, self.label_height, int(self.label_height / self.tiling)+1):
                i += 1
                x_flip = 1
                y_flip = 1
                if i%2 == 0:
                    y_flip = -1
                if j%2 == 0:
                    x_flip = -1
                flipped_pixmap = self.tile_pixmap.transformed(QTransform().scale(x_flip, y_flip))
                painter.drawPixmap(x, y, flipped_pixmap)
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
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.generator = Generator(self.image_size, self.latent_dim, feature_g=64, channels=self.image_channels)
            state_dict = torch.load(self.model_path, map_location=device)
            self.generator.load_state_dict(state_dict)
            #self.generator = torch.compile(self.generator)
            self.generator.eval()
            print(f"✅ Reloaded generator from {self.model_path}")
            print(f"✅ Running on {device}.")
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
        if self.stream != None:
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
