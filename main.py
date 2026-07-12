import math
import random
import sys
import time

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
import torch
from PIL import Image

from ncnn_vulkan import upscale_image

from models.conditional.sle_conditional import Generator
from models.upscale.sle import UGenerator
from models.conditional.sle_conditional import NoiseInjection

import midi
import stylesheets

from fullscreen import FullscreenImageWindow
from midi_dialog import MidiDialog
from models_dialog import ModelsDialog
from options_dialog import OptionsDialog
from input_dialog import InputDialog
from recording import get_sample, open_stream, push_latent


torch.set_num_interop_threads(16)
torch.set_num_threads(16)

def fake_hue_shift(img, shift):
    r, g, b = img[...,0], img[...,1], img[...,2]
    shift = int(shift)
    return np.stack([
        np.roll(r, shift, axis=1),
        np.roll(g, shift, axis=0),
        b
    ], axis=-1)



class GANVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setStyleSheet(stylesheets.stylesheet)
        self.setWindowTitle("Audio GAN Visualizer (Qt)")
        self.label_width, self.label_height = 1000, 1000
        self.resize(self.label_width, self.label_height + 55)

        # --- Central Widget ---
        self.label = QLabel()
        self.label.setStyleSheet(f"""
            QLabel {{
                background-color: {stylesheets.BG1}
            }}
        """)
        self.label.setAlignment(Qt.AlignCenter)
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 0, 10, 10)
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

        midi_device_settings = self.settings.value("midi_device", defaultValue=None)
        if midi_device_settings is not None:
            print("Found saved MIDI device.")
            self.midi_device = midi_device_settings
        else:
            self.midi_device = None

        if self.midi_device is None:
            midi_settings = self.settings.value("midi_settings", defaultValue=None)
            if midi_settings is not None:
                print("Found saved MIDI settings.")
                for i in midi_settings:
                    midi.settings[i][0] = midi_settings[i][0]
        else:
            midi_settings = self.settings.value(f"midi_settings.{self.midi_device}", defaultValue=None)
            if midi_settings is not None:
                print("Found saved MIDI settings with device.")
                for i in midi_settings:
                    try:
                        midi.settings[i][0] = midi_settings[i][0]
                        midi.settings[i][3] = midi_settings[i][3]
                        midi.settings[i][4] = midi_settings[i][4]
                    except Exception as e:
                        print(f"Failed loading: {e}")


        # --- Audio and Model Setup ---
        self.blocksize = 512
        self.latent_dim = 256
        self.image_size = 512
        self.layer = 6
        self.image_channels = 3
        self.model_path = "./models/conditional/512,512,3/Psychart-G-512-Epoch-1276.pth"
        #self.model_path = "./models/FastGAN/all_45000.pth"
        self.models = {"sle", "sle_conditional", "onnx"}
        self.model = "sle_conditional"

        self.session = None
        self.torch_device = None

        self.compile_model = False

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
        self.spectrum = np.zeros(int(self.blocksize / 2) + 1)
        self.lookup = np.arange(self.latent_dim)
        self.a = torch.randn(1, self.latent_dim)
        self.direction = torch.randn(1, self.latent_dim)
        self.high_direction = torch.randn(1, self.latent_dim).numpy()
        self.fixed_noise = torch.zeros(1, self.latent_dim)
        self.drift = 0
        self.step = 0

        self.high_pass_max = 0.0
        self.smoothed_high_pass_max = 0.0

        self.dimensions = [8,16,32,64,128,256,512]
        self.noises = {}
        for i in self.dimensions:
            self.noises[i] = torch.randn(1,1,i,i)

        # --- Onnx ---

        # --- Start Audio Stream ---
        self.stream = None
        self.open_stream()

        # --- Timer for updates ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(17)

        # --- Timer for FPS ---

        self.fps_timer = time.perf_counter()

        # --- Open Midi Device --- 
        self.midi_worker = midi.Worker(self)
        if self.midi_device is not None:
            self.open_midi_device()
        else:
            self.midi = None

        # --- Menu ---
        self.init_menu()




    def open_stream(self):
        self.stream = open_stream(self.device, self.blocksize)

    def open_midi_device(self):
        self.midi = self.midi_worker.open_midi_device(self.midi_device, midi.settings)


    def init_menu(self):
        menubar = self.menuBar()
        options_menu = menubar.addMenu("Options")

        adjust_action = QAction("Adjust Sliders", self)
        adjust_action.triggered.connect(self.open_options_dialog)
        options_menu.addAction(adjust_action)

        load_action = QAction("Model Settings", self)
        load_action.triggered.connect(self.open_models_dialog)
        options_menu.addAction(load_action)

        input_action = QAction("Input Settings", self)
        input_action.triggered.connect(self.open_input_dialog)
        options_menu.addAction(input_action)

        input_action = QAction("MIDI Settings", self)
        input_action.triggered.connect(self.open_midi_dialog)
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

    def open_midi_dialog(self):
        dialog = MidiDialog(self, self.midi_worker)
        dialog.setModal(False)
        dialog.show()

    def open_fullscreen_window(self):
        self.fullscreen = FullscreenImageWindow(self.current_pixmap)

    def update_frame(self):
        b = time.perf_counter()
        delta_t = b-self.fps_timer
        print(f"{1/delta_t:.1f}fps")

        self.fps_timer = b

        # --- Flux ---
        r_spectrum = self.spectrum

        # --- Smooth Spectrum ---
        self.step += 1
        if self.stream != None:
            self.spectrum = get_sample(
                self.stream, self.smoothed_spectrum, self.blocksize
            )
        else:
            self.spectrum = np.zeros(int(self.blocksize / 2) + 1)


        self.smoothing_factor = midi.settings["Smoothing Factor"][0]
        smoothing = 1.0 - np.exp(-delta_t*10 / max(self.smoothing_factor, 1e-6))
        self.smoothed_spectrum = smoothing * self.smoothed_spectrum + (1 - smoothing) * self.spectrum


        # --- Randomize Latent Vector ---
        if midi.settings["Audio Randomization"][0] != 0 and self.step % int(30 / midi.settings["Audio Randomization"][0]) == 0:
            c = random.randint(0, self.latent_dim - 1)
            d = random.randint(0, self.latent_dim - 1)
            self.lookup[d], self.lookup[c] = self.lookup[c], self.lookup[d]


        # --- Low Pass Drift: make big changes---
        # --- Flux ---
        low_pass_normal = self.smoothed_spectrum.max()
        low_pass_normal = max(0, low_pass_normal)
        flux = (self.spectrum - r_spectrum) / max(delta_t, 1e-6)


        low_pass_drift = np.max(flux)
        low_pass_drift = max(low_pass_drift, 0)

        low_pass = np.power(low_pass_drift, midi.settings["Lowpass Power"][0]) * 0.0001
        low_pass_drift = low_pass * midi.settings["Lowpass Sensivity"][0]
        self.drift += low_pass_drift * delta_t
        self.a = push_latent(self.a, self.direction, low_pass_drift * delta_t)




        if self.drift >= 1.0:
            self.drift = 0 
            self.direction = torch.randn(1, self.latent_dim)



        # High Pass Snare and Hi Hat Detection
        self.high_pass_max = np.max(flux[int(midi.settings["Lowpass Cutoff"][0]):len(flux)])

        self.high_pass_max = max(0, self.high_pass_max)
        self.smoothed_high_pass_max = self.smoothed_high_pass_max * (smoothing) + self.high_pass_max * (1 - smoothing)







        spectrum = torch.zeros(1, self.latent_dim)
        for i in range(self.latent_dim):
            spectrum[0][i] = self.smoothed_spectrum[self.lookup[i]] * (1 + 0)

        noise = torch.zeros(1, self.latent_dim)
        cutoff = int(midi.settings["Cutoff"][0])
        audio_noise = spectrum * midi.settings["Audio Weight"][0]
        audio_noise = torch.cat([self.fixed_noise[0,:cutoff],audio_noise[0,cutoff:]]).view(1,-1)
        for i in range(self.latent_dim):
            noise[0][i] = self.a[0][i] * midi.settings["Noise Weight"][0] + audio_noise[0][i]


        noise = noise.view(1,self.latent_dim,1,1)



        if (self.model == "sle" or self.model == "sle_conditional"):
            noise.to(self.torch_device)
            for m in self.generator.modules():
                if isinstance(m, NoiseInjection):
                    m.set_noise(self.noises[m.size] * (midi.settings["Noise Base"][0] + self.smoothed_high_pass_max * midi.settings["Noise Injection"][0]))
            with torch.no_grad():
                if self.model == "sle_conditional":
                    red = midi.settings["X"][0] + midi.settings["Y"][0] * low_pass_normal * 0.02
                    y = torch.tensor([red], dtype=torch.float32)
                    y.to(self.torch_device)
                    image = self.generator(noise,y,True)[0]
                else:
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
        image = np.clip(image, 0, 1)
        image_array = (image * 255).astype(np.uint8)
        if self.image_channels == 1:
            image_array = np.stack([image_array] * 3, axis=0)
        image_rgb = np.transpose(image_array, (1, 2, 0))


        # Hue Shift and Shake
        image_rgb = fake_hue_shift(image_rgb, low_pass * midi.settings["Hue Shift"][0] * 10)

        image_rgb = np.ascontiguousarray(image_rgb)




        pil_img = Image.fromarray(image_rgb, mode="RGB")
        #pil_img = pil_img.resize((128,128))
        
        pil_img = upscale_image(pil_img, scale=4)
        image_rgb = np.asarray(pil_img, dtype=np.uint8)


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
            self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            if (self.model == "onnx"):
                print(f"❌ No Onnx Support")

            if (self.model == "sle"):
                self.generator = UGenerator(nz=self.latent_dim, ngf=16, nc=3, img_size=self.image_size, layer=self.image_size)
                state_dict = torch.load(self.model_path, map_location=self.torch_device)
                self.generator.load_state_dict(state_dict)
                print(f"✅ Reloaded Upscale Generator from {self.model_path}") 
                self.generator.eval()
                if self.compile_model:
                    self.generator = torch.compile(self.generator)
                self.generator.to(self.torch_device)
                print(f"✅ Running on {self.torch_device}.")

            if (self.model == "sle_conditional"):
                self.generator = Generator(nz=self.latent_dim, ngf=8, nc=3, img_size=self.image_size, num_classes=1, skip_layer=3)
                state_dict = torch.load(self.model_path, map_location=self.torch_device)
                self.generator.load_state_dict(state_dict)
                print(f"✅ Reloaded Upscale Conditional Generator from {self.model_path}")
                self.generator.eval()
                if self.compile_model:
                    self.generator = torch.compile(self.generator)
                self.generator.to(self.torch_device)
                print(f"✅ Running on {self.torch_device}.")
                print("Coming Soon")

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
        if self.midi_device is None:
            self.settings.setValue("midi_settings", midi.settings)
        else:
            self.settings.setValue(f"midi_settings.{self.midi_device}", midi.settings)
            self.settings.setValue("midi_device", self.midi_device)
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
