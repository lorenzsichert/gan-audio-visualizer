
BG1 = "#000000"
BG2 = "#222222"
BG3 = "#333333"
BG5 = "#555555"
BG7 = "#777777"
BG9 = "#999999"
TEXT_COLOR = "#FFFFFF"
ACCENT_COLOR = "#999999"

stylesheet = f"""
    QMainWindow {{
        background-color: {BG1};
        color: {TEXT_COLOR};
        padding: 0px;
    }}
    QLabel {{
        background-color: {BG2};
        color: {TEXT_COLOR};
    }}

    QDialog {{
        background-color: {BG2};
        color: {TEXT_COLOR};
    }}
    QPushButton{{ 
        background-color: {BG5};
        color: {TEXT_COLOR};
        padding: 6px 10px;
        border-radius: 6px;
    }}
    QDoubleSpinBox {{
        background-color: {BG3};
        color: {TEXT_COLOR};
        border-radius: 2px;
        width: 100px;
        padding: 4px;
    }}
    QSpinBox {{
        background-color: {BG3};
        color: {TEXT_COLOR};
        border-radius: 2px;
        padding: 4px;
    }}

    /* -------------- Slider ---------------- */
    QSlider::groove:horizontal {{
        height: 16px;
        background: #444444;
        border-radius: 6px;
    }}
    QSlider::sub-page:horizontal {{
        background: {ACCENT_COLOR};
        border-radius: 8px;
    }}
    QSlider::add-page:horizontal {{
        background: {BG3};
        border-radius: 8px;
    }}
    QSlider::handle:horizontal {{
        background: {BG5};
        border: 1px solid {BG9};
        width: 24px;
        height: 24px;
        margin: -4px -4px;
        border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {BG7};
    }}
    QSlider::handle:horizontal:pressed {{
        background: {ACCENT_COLOR};
    }}
    /* ---------------- Menu ----------------- */
    QMenu {{
        background-color: {BG2};
        color: {TEXT_COLOR};
        padding: 8px;
        border-radius: 6px;
    }}
    QMenu::item {{
        background-color: {BG2};
        color: {TEXT_COLOR};
        padding: 6px 10px;
        border-radius: 6px;
    }}
    QMenu::item:selected {{
        background-color: {BG7};
    }}
    QMenuBar {{
        background-color: {BG1};
        color: {TEXT_COLOR};
        padding: 10px;
    }}
    QMenuBar::item {{
        background-color: {BG2};
        color: {TEXT_COLOR};
        padding: 8px;
        border-radius: 6px;
    }}
    QMenuBar::item:selected {{
        background-color: {BG7};
    }}


    QTabWidget::pane {{
        background: {BG2};
        border: 1px solid {BG5};
        border-radius: 8px;
    }}

    QTabBar::tab {{
        background: {BG3};
        color: {TEXT_COLOR};
        padding: 8px 16px;
        margin: 2px;
        border-radius: 8px;
    }}

    QTabBar::tab:selected {{
        background: {ACCENT_COLOR};
        color: {TEXT_COLOR};
    }}

    QTabBar::tab:hover {{
        background: {BG5};
    }}

    QTabBar::tab:!selected {{
        margin-top: 4px;  /* slight depth effect */
    }}


    QCheckBox {{
        color: {TEXT_COLOR};
        spacing: 8px;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 2px solid {BG5};
        border-radius: 4px;
        background-color: {BG3};
    }}
    QCheckBox::indicator:hover {{
        border: 2px solid {BG7};
    }}
    QCheckBox::indicator:checked {{
        background-color: {ACCENT_COLOR};
        border: 2px solid #FFFFFF;
    }}

    QLineEdit {{
        background-color: {BG3};
        color: {TEXT_COLOR};
        border: 1px solid {BG5};
        border-radius: 8px;
        padding: 6px 10px;
        selection-background-color: {ACCENT_COLOR};
        selection-color: {BG1};
    }}

    /* Hover */
    QLineEdit:hover {{
        border: 1px solid {BG7};
    }}

    /* Focus (important) */
    QLineEdit:focus {{
        border: 1px solid {ACCENT_COLOR};
        background-color: {BG2};
    }}

    /* .--------------- Folder ----------------- */



    QTreeView, QListView {{
        background-color: {BG2};
        color: {TEXT_COLOR};
        border: 1px solid {BG5};
        border-radius: 8px;
    }}

    /* Folder items */
    QTreeView::item {{
        padding: 6px;
    }}

    QTreeView::item:hover {{
        background-color: {BG3};
    }}

    QTreeView::item:selected {{
        background-color: {ACCENT_COLOR};
        color: {BG1};
        border-radius: 6px;
    }}

    /* Header */
    QHeaderView::section {{
        background-color: {BG3};
        color: {TEXT_COLOR};
        padding: 6px;
        border: none;
    }}

    /* ===== Scrollbars ===== */

    /* Vertical scrollbar */
    QScrollBar:vertical {{
        background: {BG2};
        width: 12px;
        margin: 0px;
        border-radius: 6px;
    }}

    QScrollBar::handle:vertical {{
        background: {BG5};
        min-height: 20px;
        border-radius: 6px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: {ACCENT_COLOR};
    }}

    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    /* Horizontal scrollbar */
    QScrollBar:horizontal {{
        background: {BG2};
        height: 12px;
        margin: 0px;
        border-radius: 6px;
    }}

    QScrollBar::handle:horizontal {{
        background: {BG5};
        min-width: 20px;
        border-radius: 6px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background: {ACCENT_COLOR};
    }}

    QScrollBar::add-line:horizontal,
    QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}

    /* Main Combo Box */
    QComboBox {{
        background-color: {BG3};
        color: {TEXT_COLOR};
        border: 1px solid {BG5};
        border-radius: 8px;
        padding: 6px 10px;
    }}

    QComboBox:hover {{
        border: 1px solid {BG7};
    }}

    QComboBox:focus {{
        border: 1px solid {ACCENT_COLOR};
        background-color: {BG2};
    }}

    /* Dropdown arrow area */
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}

    /* Optional: custom arrow */
    QComboBox::down-arrow {{
        image: none;
        width: 10px;
        height: 10px;
    }}

    /* Dropdown list */
    QComboBox QAbstractItemView {{
        background-color: {BG2};
        color: {TEXT_COLOR};
        border: 1px solid {BG5};
        selection-background-color: {ACCENT_COLOR};
        selection-color: {BG1};
        border-radius: 8px;
        padding: 4px;
    }}
"""
