from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QDialogButtonBox,
    QFileDialog, QLineEdit, QPushButton, QSpinBox,
    QLabel,
)


class ModelsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Load Models")
        self.parent = parent

        layout = QFormLayout()

        # --- Model file path ---
        self.model_path_edit = QLineEdit(self)
        self.model_path_edit.setText(self.parent.model_path)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_model)
        layout.addRow("Model Path:", self.model_path_edit)
        layout.addRow("", browse_btn)

        # --- Image size selector ---
        self.size_box = QSpinBox()
        self.size_box.setRange(16, 512)
        self.size_box.setValue(self.parent.image_size)
        layout.addRow("Image Size:", self.size_box)

        # --- Channel selector ---
        self.channel_box = QSpinBox()
        self.channel_box.setRange(1, 3)
        self.channel_box.setValue(self.parent.image_channels)
        layout.addRow("Channels:", self.channel_box)

        # --- Reload button ---
        reload_btn = QPushButton("Reload Generator")
        reload_btn.clicked.connect(self.reload_generator)
        layout.addRow(reload_btn)

        # Close button
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def browse_model(self):
        """Open file dialog to choose a model file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "models/",
            "PyTorch Models (*.pth *.pt)"
        )
        if path:
            self.model_path_edit.setText(path)

    def reload_generator(self):
        """Reload the generator in the main window with selected params."""
        new_path = self.model_path_edit.text()
        new_size = self.size_box.value()
        new_channels = self.channel_box.value()

        # Update parent attributes
        self.parent.model_path = new_path
        self.parent.image_size = new_size
        self.parent.image_channels = new_channels

        # Call parent's reload function
        self.parent.reload_generator()

        # Optional feedback
        QLabel("Model reloaded!").show()

    def update_image_size(self, value):
        self.parent.image_size = value

    def update_channels(self, value):
        self.parent.image_channels = value 
 
    def update_model_path(self, value):
        self.parent.model_path = value
