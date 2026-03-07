from PyQt5.QtWidgets import (
    QDialog, QFormLayout, 
    QFileDialog, QLineEdit, QPushButton, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget,
)

from pathlib import Path
import re

import stylesheets


class ModelsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Load Models")
        self.setMinimumSize(700,500)
        self.setStyleSheet(stylesheets.stylesheet)
        self.parent = parent
        self.path = ""


        layout = QVBoxLayout()

        tabs = QTabWidget()
        layout.addWidget(tabs)

        # ---- Tab 1 ---
        tab1 = QWidget()
        tab_custom_models = QFormLayout(tab1)


        # --- Model file path ---
        self.model_path_edit = QLineEdit(self)
        self.model_path_edit.setText(self.parent.model_path)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(lambda: self.browse_custom_model("models/","PyTorch Files (*.pth)"))
        tab_custom_models.addRow("Model Path:", self.model_path_edit)
        tab_custom_models.addRow("", browse_btn)

        # --- Image size selector ---
        self.size_box = QSpinBox()
        self.size_box.setRange(16, 512)
        self.size_box.setValue(self.parent.image_size)
        tab_custom_models.addRow("Image Size:", self.size_box)

        # --- Channel selector ---
        self.channel_box = QSpinBox()
        self.channel_box.setRange(1, 3)
        self.channel_box.setValue(self.parent.image_channels)
        tab_custom_models.addRow("Channels:", self.channel_box)

        # --- Layer selector ---
        self.layer_box = QSpinBox()
        self.layer_box.setRange(1, 1000)
        self.layer_box.setValue(self.parent.layer)
        tab_custom_models.addRow("Layers:", self.layer_box)

        # --- Reload button ---
        reload_btn = QPushButton("Reload Generator")
        reload_btn.clicked.connect(self.reload_generator)
        tab_custom_models.addRow(reload_btn)


        # --- Tab 2 ----
        tab2 = QWidget()
        tab_onnx = QFormLayout(tab2)


        self.onnx_path_edit = QLineEdit(self)
        self.onnx_path_edit.setText("Select Onnx File")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(lambda: self.browse_onnx_model("onnx/","Onnx Files (*.onnx)"))
        tab_onnx.addRow("Onnx Path:", self.onnx_path_edit)
        tab_onnx.addRow("", browse_btn)

        reload_onnx_btn = QPushButton("Reload Generator")
        reload_onnx_btn.clicked.connect(self.reload_onnx)
        tab_onnx.addRow(reload_onnx_btn)



        tabs.addTab(tab1, "Custom Model")
        tabs.addTab(tab2, "Onnx Model")

        self.setLayout(layout)

    def browse_custom_model(self, path, extension):
        """Open file dialog to choose a model file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            path,
            extension
        )
        if path:
            self.model_path_edit.setText(path)
            path = Path(path)
            
            # Look for a folder name containing something like 32,32,3
            try:
                match = re.search(r'(\d+),(\d+),(\d+)', str(path))
                height, _, channels = map(int, match.groups())
                self.size_box.setValue(height)
                self.channel_box.setValue(channels)
            except Exception as e:
                print(f"Error parsing File Path: {e}")

    def browse_onnx_model(self, path, extension):
        """Open file dialog to choose a model file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            path,
            extension
        )
        if path:
            self.onnx_path_edit.setText(path)

    def reload_generator(self):
        """Reload the generator in the main window with selected params."""
        new_path = self.model_path_edit.text()
        new_size = self.size_box.value()
        new_channels = self.channel_box.value()

        # Update parent attributes
        self.parent.model_path = new_path
        self.parent.image_size = new_size
        self.parent.image_channels = new_channels
        self.parent.layer = self.layer_box.value()
        self.parent.model = "custom"

        # Call parent's reload function
        self.parent.reload_generator()

    def reload_onnx(self):
        self.parent.latent_dim = 256
        self.onnx_path = self.onnx_path_edit.text()
        self.parent.model_path = self.onnx_path
        self.parent.model = "onnx"
        self.parent.reload_generator()

    def update_image_size(self, value):
        self.parent.image_size = value

    def update_channels(self, value):
        self.parent.image_channels = value 
 
    def update_model_path(self, value):
        self.parent.model_path = value
