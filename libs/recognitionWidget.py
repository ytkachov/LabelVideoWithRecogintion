
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from libs.threading import *
from libs.objectDetector import *
import time
import pickle


ADD_MODEL_COMMAND = '<< Add Model >>'
RUN_DETECTION = 'RunDetection'
MODEL_LIST = 'ModelList'
CURRENT_MODEL_NAME = 'CurrentModelName'

class Recognition(QDockWidget):

    def __init__(self, Title, Settings, ParentWindow):
        super(Recognition, self).__init__(Title, ParentWindow)

        settings = {RUN_DETECTION: True, MODEL_LIST: [], CURRENT_MODEL_NAME: ''}
        if Settings:
            settings = pickle.loads(Settings)

        self.threadpool = QThreadPool()

        self.objectDetectors = {}
        self.currentObjectDetector = None

        self.runDetection = settings.get(RUN_DETECTION, False)
        self.modelList = settings.get(MODEL_LIST, [])
        self.currentModelName = settings.get(CURRENT_MODEL_NAME, '')

        self.setFloating(False)
        self.setAllowedAreas(Qt.TopDockWidgetArea)
        self.setFeatures(QDockWidget.NoDockWidgetFeatures)
        ParentWindow.addDockWidget(Qt.TopDockWidgetArea, self)

        self.runDetectionCheckbox = QCheckBox('Detect objects')
        self.runDetectionCheckbox.setChecked(self.runDetection)
        self.runDetectionCheckbox.stateChanged.connect(self._run_detection_changed)

        self.detectionModelsCombobox = QComboBox(self)
        self.detectionModelsCombobox.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.detectionModelsCombobox.activated.connect(self._model_selected)

        self.detectionModelsCombobox.addItems(self.modelList)
        self.detectionModelsCombobox.addItem(ADD_MODEL_COMMAND)
        currentModelIdx = 0
        idx = 0
        for mn in self.modelList:
            if mn == self.currentModelName:
                currentModelIdx = idx
                break

            idx += 1

        self.detectionModelsCombobox.setCurrentIndex(currentModelIdx)
        self._update_model(self.detectionModelsCombobox.itemText(currentModelIdx))

        self.detectionModelsCombobox.setVisible(self.runDetection)

        detectionLayout = QHBoxLayout()
        detectionLayout.addWidget(self.runDetectionCheckbox)
        detectionLayout.addWidget(self.detectionModelsCombobox)
        detectionLayout.addStretch()

        detectionContainer = QWidget()
        detectionContainer.setLayout(detectionLayout)
        self.setWidget(detectionContainer)

        if self.detectionModelsCombobox.currentText() != ADD_MODEL_COMMAND:
            self.currentModelName = self.detectionModelsCombobox.currentText()

    def Settings(self):
        settings = {RUN_DETECTION: self.runDetection, MODEL_LIST: self.modelList, CURRENT_MODEL_NAME: self.currentModelName}
        return pickle.dumps(settings)

    def _run_detection_changed(self, item= None):
        self.runDetection = self.runDetectionCheckbox.isChecked()
        self.detectionModelsCombobox.setVisible(self.runDetection)

    def _model_selected(self, idx):
        # remember current model index
        for i in range(self.detectionModelsCombobox.count() - 1):
            if self.currentModelName == self.detectionModelsCombobox.itemText(i):
                selectedModel = i
                break

        if idx == self.detectionModelsCombobox.count() - 1:
            # add new model
            filename = QFileDialog.getOpenFileName(None, caption='Open model', directory='.',
                                                   filter='TensorFlow model file (*.pb)')[0]
            if filename:
                selectedModel = -1
                models = []
                for i in range(self.detectionModelsCombobox.count() - 1):
                    model = self.detectionModelsCombobox.itemText(i)
                    models.append(model)
                    if filename == model:
                        selectedModel = i

                if selectedModel == -1:
                    models.append(filename)
                    self.modelList.append(filename)
                    models.append(ADD_MODEL_COMMAND)

                    self.detectionModelsCombobox.clear()
                    self.detectionModelsCombobox.addItems(models)
                    selectedModel = self.detectionModelsCombobox.count() - 2

            self.detectionModelsCombobox.setCurrentIndex(selectedModel)
        else:
            selectedModel = idx

        self._update_model(self.detectionModelsCombobox.itemText(selectedModel))

    def _update_model(self, newModelName):
        if newModelName != ADD_MODEL_COMMAND and newModelName != '':
            self.currentModelName = newModelName
            if self.currentModelName in self.objectDetectors:
                self.currentObjectDetector = self.objectDetectors[self.currentModelName]
                return

            self.detectionModelsCombobox.setEnabled(False)
            self.runDetectionCheckbox.setEnabled(False)

            #load model in separate thread
            worker = Worker(self._load_model, newModelName)  # Any other args, kwargs are passed to the run function
            worker.signals.result.connect(self._on_loading_result)
            worker.signals.error.connect(self._on_loading_error)
            worker.signals.finished.connect(self._on_loading_finished)

            # Execute
            self.threadpool.start(worker)

    def _on_loading_result(self, objectDetector):
        if objectDetector != None:
            self.objectDetectors[self.currentModelName] = objectDetector
            self.currentObjectDetector = objectDetector

    def _on_loading_finished(self):
        self.detectionModelsCombobox.setEnabled(True)
        self.runDetectionCheckbox.setEnabled(True)

    def _load_model(self, modelName):
        od = ObjectDetector(modelName, r'C:\venv\models\research\object_detection\faster_rcnn_inception_v2_excavator_1\excavator_labelmap.pbtxt')
        od.Detect(10)

        return od

    def _on_loading_error(self, restuple):
        str = restuple[2]