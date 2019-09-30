from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class TrainingSettings(QDialog):
    __dialogWidth = 400

    def __init__(self, parent, src_model_path="", train_model_path="", graph_path=""):
        super(TrainingSettings, self).__init__(parent)

        # source model
        model_path_label = QLabel("Path to source model: ")
        self._model_path_edit = QLineEdit(src_model_path)
        model_path_button = QPushButton("...")
        width = model_path_button.fontMetrics().boundingRect("...").width() + 12
        model_path_button.setMaximumWidth(width)
        model_path_button.clicked.connect(self._on_model_path)

        model_path_layout = QHBoxLayout()
        model_path_layout.setContentsMargins(0, 0, 0, 0)
        model_path_layout.addWidget(model_path_label)
        model_path_layout.addWidget(self._model_path_edit)
        model_path_layout.addWidget(model_path_button)
        model_path_widget = QWidget()
        model_path_widget.setLayout(model_path_layout)

        # trained model
        trained_model_label = QLabel("Path to training data: ")
        self._trained_model_edit = QLineEdit(train_model_path)
        trained_model_button = QPushButton("...")
        width = trained_model_button.fontMetrics().boundingRect("...").width() + 12
        trained_model_button.setMaximumWidth(width)
        trained_model_button.clicked.connect(self._on_trained_model)

        trained_model_layout = QHBoxLayout()
        trained_model_layout.setContentsMargins(0, 0, 0, 0)
        trained_model_layout.addWidget(trained_model_label)
        trained_model_layout.addWidget(self._trained_model_edit)
        trained_model_layout.addWidget(trained_model_button)
        trained_model_widget = QWidget()
        trained_model_widget.setLayout(trained_model_layout)

        # inference graph
        graph_label = QLabel("Inference graph: ")
        self._graph_edit = QLineEdit(graph_path)
        graph_button = QPushButton("...")
        width = graph_button.fontMetrics().boundingRect("...").width() + 12
        graph_button.setMaximumWidth(width)
        graph_button.clicked.connect(self._on_graph)

        graph_layout = QHBoxLayout()
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.addWidget(graph_label)
        graph_layout.addWidget(self._graph_edit)
        graph_layout.addWidget(graph_button)
        graph_widget = QWidget()
        graph_widget.setLayout(graph_layout)


        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self._on_ok)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self._on_cancel)
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.addStretch()
        buttons_layout.addWidget(ok_button)
        buttons_layout.addWidget(cancel_button)
        buttons_widget = QWidget()
        buttons_widget.setLayout(buttons_layout)

        # Create layout and add widgets
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        layout.addWidget(model_path_widget)
        layout.addWidget(trained_model_widget)
        layout.addWidget(graph_widget)
        layout.addWidget(buttons_widget)

        # Set dialog layout
        self.setLayout(layout)
        self.adjustSize()
        self.setMinimumWidth(TrainingSettings.__dialogWidth)
        self.setFixedHeight(self.height())
        self.setWindowTitle("Training settings")

        # Add button signal to greetings slot

        self.retval = False
        self.result = (src_model_path, train_model_path, graph_path)

    def _on_ok(self):
        TrainingSettings.__dialogWidth = self.width()
        self.result = (self._model_path_edit.text(), self._trained_model_edit.text(), self._graph_edit.text())
        self.retval = True
        self.accept()

    def _on_cancel(self):
        TrainingSettings.__dialogWidth = self.width()
        self.retval = False
        self.accept()

    def _on_model_path(self):
        path = self._model_path_edit.text()
        new_path = QFileDialog.getExistingDirectory(self, 'Select existing model folder', path if path else './')
        if new_path:
            self._model_path_edit.setText(new_path)

    def _on_trained_model(self):
        path = self._trained_model_edit.text()
        new_path = QFileDialog.getExistingDirectory(self, 'Select trainning model folder', path if path else './')
        if new_path:
            self._trained_model_edit.setText(new_path)

    def _on_graph(self):
        path = self._model_path_edit.text()
        new_path = QFileDialog.getExistingDirectory(self, 'Select inference graph folder', path if path else './')
        if new_path:
            self._graph_edit.setText(new_path)

    def getResult(self):
        return self.result

    def exec_(self):
        super(TrainingSettings, self).exec_()
        return self.retval