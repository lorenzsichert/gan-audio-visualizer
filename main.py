import random
import sys

from PyQt5.QtCore import QSettings, QTimer, Qt
from PyQt5.QtGui import (
    QImage, 
    QPainter, 
    QPixmap, 
    QTransform
)

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
from numpy.random import randint, randn
import torch

from fastgan_models import FastGenerator
import onnxruntime as ort

from fullscreen import FullscreenImageWindow
import midi
from training.models import Generator
from models_dialog import ModelsDialog
from options_dialog import OptionsDialog
from input_dialog import InputDialog
from recording import get_sample, open_stream, push_latent



def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


class GANVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audio GAN Visualizer (Qt)")
        self.label_width, self.label_height = 900, 900
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

        midi_settings = self.settings.value("midi_settings", defaultValue=None)
        if midi_settings is not None:
            print("Found saved MIDI settings.")
            for i in midi.settings:
                midi.settings[i][0] = midi_settings[i][0]

        # --- Audio and Model Setup ---
        self.blocksize = 512
        self.latent_dim = 256
        self.image_size = 256
        self.layer = 6
        self.image_channels = 3
        self.model_path = "onnx/fastgan.onnx"
        #self.model_path = "./models/FastGAN/all_5000.pth"
        self.model = "onnx"

        self.session = None

        self.reload_generator()

        # --- Tiling Settings ---
        self.tiling = 1
        self.current_pixmap = QPixmap(self.label_width, self.label_height)
        self.tile_pixmap = QPixmap(self.label_width // self.tiling,
                                   self.label_height // self.tiling)

        # --- Fullscreen Window ---
        self.fullscreen = None

        # --- Parameters ---
        self.smoothing_factor = 0.55
        self.noise_weight = 1.0
        self.audio_weight = 0.1
        self.noise_randomization = 10
        self.audio_randomization = 4


        self.smoothed_spectrum = np.zeros(int(self.blocksize / 2) + 1)
        self.lookup = np.arange(self.latent_dim)
        self.a = torch.randn(1, self.latent_dim)
        self.b = torch.randn(1, self.latent_dim)
        self.direction = torch.randn(1, self.latent_dim)
        self.drift = 0
        self.step = 0

        # --- Onnx ---

        # --- Start Audio Stream ---
        self.stream = None
        self.open_stream()

        # --- Timer for updates ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

        # --- Open Midi Device --- 
        self.midi = midi.open_midi_device(midi.settings)


        # --- Menu ---
        self.init_menu()




    def open_stream(self):
        self.stream = open_stream(self.device, self.blocksize)


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

        fullscreen_action = QAction("Open Fullscreen Window", self)
        fullscreen_action.triggered.connect(self.open_fullscreen_window)
        options_menu.addAction(fullscreen_action)

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

    def open_fullscreen_window(self):
        self.fullscreen = FullscreenImageWindow(self.current_pixmap)

    def update_frame(self):
        # --- Smooth Spectrum ---
        self.step += 1
        if self.stream != None:
            self.smoothed_spectrum = get_sample(
                self.stream, self.smoothed_spectrum, self.blocksize, midi.settings["Smoothing Factor"][0]
            )
        else:
            self.smoothed_spectrum = np.zeros(int(self.blocksize / 2) + 1)

        # --- Randomize Latent Vector ---
        if midi.settings["Audio Randomization"][0] != 0 and self.step % int(30 / midi.settings["Audio Randomization"][0]) == 0:
            c = random.randint(0, self.latent_dim - 1)
            d = random.randint(0, self.latent_dim - 1)
            self.lookup[d], self.lookup[c] = self.lookup[c], self.lookup[d]
            index = randint(0,self.latent_dim)
            self.b[0][index] = randn()


        # --- Low Pass Drift: make big changes---
        low_pass_drift = 0.0
        for i in range(int(midi.settings["Lowpass Cutoff"][0])):
            low_pass_drift = max(self.smoothed_spectrum[i],low_pass_drift)

        low_pass_drift = np.pow(low_pass_drift, midi.settings["Lowpass Power"][0])
        low_pass_drift *= midi.settings["Lowpass Sensivity"][0] * 0.0001
        self.drift += low_pass_drift
        self.a = push_latent(self.a, self.direction, low_pass_drift)


        if self.drift >= 1.0:
            self.drift = 0 
            self.direction = torch.randn(1, self.latent_dim)


        spectrum = np.zeros(self.latent_dim)
        for i in range(self.latent_dim):
            spectrum[i] = self.smoothed_spectrum[self.lookup[i]]

        noise = torch.zeros(1, self.latent_dim)
        for i in range(self.latent_dim):
            noise[0][i] = self.a[0][i] * midi.settings["Noise Weight"][0] + spectrum[i] * midi.settings["Audio Weight"][0] * self.b[0][i]

        noise = noise.view(1,self.latent_dim,1,1)

        if (self.model == "custom"):
            image = self.generator(noise, 1.0).detach().squeeze()
            image = image.numpy()
        if (self.model == "fastgan"):
            with torch.no_grad():
                image = self.generator(noise)[0]
            image = image.numpy()


        if (self.model == "onnx"):
            inputs = {
                "z": noise.numpy(),
            }
            image = self.session.run(['image'], inputs)[0][0]
            

        image = np.clip(image, -1, 1)
        image = (image + 1) / 2.0
        #image = np.where(image < 0.25, 0, image)
        image_array = (image * 255).astype(np.uint8)
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
        if self.fullscreen and self.fullscreen.isVisible() and self.current_pixmap:
            self.fullscreen.update_pixmap(self.current_pixmap)
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

            if (self.model == "custom"):
                self.generator = Generator(4, self.latent_dim, self.image_size, 64,3,self.layer)
                state_dict = torch.load(self.model_path, map_location=device)
                self.generator.load_state_dict(state_dict)
                print(f"✅ Reloaded generator from {self.model_path}")
                #self.generator = torch.compile(self.generator)
                self.generator.to(device)
                self.generator.eval()
                print(f"✅ Running on {device}.")

            if (self.model == "onnx"):
                # Onnx Optimization Options
                sessionOptions = ort.SessionOptions()
                sessionOptions.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(self.model_path,sessionOptions)

                print(f"Available Providers: {self.session.get_provider_options()}")
                print(f"✅ Reloaded ONNX BigGAN from {self.model_path}")
                print(f"✅ Running on {self.session.get_providers()}.")

            if (self.model == "fastgan"):
                self.generator = FastGenerator(ngf=64,nz=256,nc=3, im_size=self.image_size)
                state_dict = torch.load(self.model_path, map_location=device)
                load_params(self.generator, state_dict["g_ema"])
                print(f"✅ Reloaded FastGAN generator from {self.model_path}")
                #self.generator = torch.compile(self.generator)
                self.generator.to(device)
                #self.generator.eval()
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
        self.settings.setValue("midi_settings", midi.settings)
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
