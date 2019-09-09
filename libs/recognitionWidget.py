
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from libs.threading import *
from libs.objectDetector import *
from libs.detectedShape import *
import time
import pickle


ADD_MODEL_COMMAND = '<< Add Model >>'
RUN_DETECTION = 'RunDetection'
MODEL_LIST = 'ModelList'
CURRENT_MODEL_NAME = 'CurrentModelName'

class Recognition(QDockWidget):

    def __init__(self, Title, Settings, ParentWindow):
        super(Recognition, self).__init__(None, ParentWindow)
        self.setWindowFlags(Qt.FramelessWindowHint)

        settings = {RUN_DETECTION: True, MODEL_LIST: [], CURRENT_MODEL_NAME: ''}
        if Settings:
            settings = pickle.loads(Settings)

        self._threadPool = QThreadPool()

        self._objectDetectors = {}
        self._currentObjectDetector = None

        self._runDetection = settings.get(RUN_DETECTION, False)
        self._modelList = settings.get(MODEL_LIST, [])
        self._currentModelName = settings.get(CURRENT_MODEL_NAME, '')

        self.setFloating(False)
        self.setAllowedAreas(Qt.TopDockWidgetArea)
        self.setFeatures(QDockWidget.NoDockWidgetFeatures | QDockWidget.DockWidgetVerticalTitleBar)
        ParentWindow.addDockWidget(Qt.TopDockWidgetArea, self)

        self._runDetectionCheckbox = QCheckBox('Detect objects')
        self._runDetectionCheckbox.setChecked(self._runDetection)
        self._runDetectionCheckbox.stateChanged.connect(self._run_detection_changed)

        self._detectionModelsCombobox = QComboBox(self)
        self._detectionModelsCombobox.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._detectionModelsCombobox.activated.connect(self._model_selected)

        self._detectionModelsCombobox.addItems(self._modelList)
        self._detectionModelsCombobox.addItem(ADD_MODEL_COMMAND)
        currentModelIdx = 0
        idx = 0
        for mn in self._modelList:
            if mn == self._currentModelName:
                currentModelIdx = idx
                break

            idx += 1

        self._detectionModelsCombobox.setCurrentIndex(currentModelIdx)
        self._update_model(self._detectionModelsCombobox.itemText(currentModelIdx))

        self._detectionModelsCombobox.setVisible(self._runDetection)

        detectionLayout = QHBoxLayout()
        detectionLayout.addWidget(self._runDetectionCheckbox)
        detectionLayout.addWidget(self._detectionModelsCombobox)
        detectionLayout.addStretch()

        detectionContainer = QWidget()
        detectionContainer.setLayout(detectionLayout)
        self.setWidget(detectionContainer)

        if self._detectionModelsCombobox.currentText() != ADD_MODEL_COMMAND:
            self._currentModelName = self._detectionModelsCombobox.currentText()

    # Public methods
    def Settings(self):
        settings = {RUN_DETECTION: self._runDetection, MODEL_LIST: self._modelList, CURRENT_MODEL_NAME: self._currentModelName}
        return pickle.dumps(settings)

    def ProcessImage(self, imgname):
        self.currentImageName = imgname
        #  run detection in separate thread
        worker = Worker(self._detect_objects, self.currentImageName)  # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self._on_detection_result)
        worker.signals.error.connect(self._on_detection_error)
        worker.signals.finished.connect(self._on_detection_finished)

        # Execute
        self._threadPool.start(worker)

    # signals
    objects_detected = pyqtSignal(tuple)

    # private methods
    def _detect_objects(self, image_path):
        return (image_path, self._currentObjectDetector.Detect(image_path))

    def _on_detection_result(self, detection_result):
        self.objects_detected.emit(detection_result)

    def _on_detection_error(self, restuple):
        str = restuple[2]

    def _on_detection_finished(self):
        self._detectionModelsCombobox.setEnabled(True)
        self._runDetectionCheckbox.setEnabled(True)

    def _run_detection_changed(self, item= None):
        self._runDetection = self._runDetectionCheckbox.isChecked()
        self._detectionModelsCombobox.setVisible(self._runDetection)

    def _load_model(self, modelName):
        od = ObjectDetector(modelName, r'C:\venv\models\research\object_detection\faster_rcnn_inception_v2_excavator_1\excavator_labelmap.pbtxt')
        return od

    def _on_loading_result(self, objectDetector):
        if objectDetector != None:
            self._objectDetectors[self._currentModelName] = objectDetector
            self._currentObjectDetector = objectDetector

    def _on_loading_finished(self):
        self._detectionModelsCombobox.setEnabled(True)
        self._runDetectionCheckbox.setEnabled(True)

    def _on_loading_error(self, restuple):
        str = restuple[2]

    def _model_selected(self, idx):
        # remember current model index
        for i in range(self._detectionModelsCombobox.count() - 1):
            if self._currentModelName == self._detectionModelsCombobox.itemText(i):
                selectedModel = i
                break

        if idx == self._detectionModelsCombobox.count() - 1:
            # add new model
            filename = QFileDialog.getOpenFileName(None, caption='Open model', directory='.',
                                                   filter='TensorFlow model file (*.pb)')[0]
            if filename:
                selectedModel = -1
                models = []
                for i in range(self._detectionModelsCombobox.count() - 1):
                    model = self._detectionModelsCombobox.itemText(i)
                    models.append(model)
                    if filename == model:
                        selectedModel = i

                if selectedModel == -1:
                    models.append(filename)
                    self._modelList.append(filename)
                    models.append(ADD_MODEL_COMMAND)

                    self._detectionModelsCombobox.clear()
                    self._detectionModelsCombobox.addItems(models)
                    selectedModel = self._detectionModelsCombobox.count() - 2

            self._detectionModelsCombobox.setCurrentIndex(selectedModel)
        else:
            selectedModel = idx

        self._update_model(self._detectionModelsCombobox.itemText(selectedModel))

    def _update_model(self, newModelName):
        if newModelName != ADD_MODEL_COMMAND and newModelName != '':
            self._currentModelName = newModelName
            if self._currentModelName in self._objectDetectors:
                self._currentObjectDetector = self._objectDetectors[self._currentModelName]
                return

            self._detectionModelsCombobox.setEnabled(False)
            self._runDetectionCheckbox.setEnabled(False)

            #load model in separate thread
            worker = Worker(self._load_model, newModelName)  # Any other args, kwargs are passed to the run function
            worker.signals.result.connect(self._on_loading_result)
            worker.signals.error.connect(self._on_loading_error)
            worker.signals.finished.connect(self._on_loading_finished)

            # Execute
            self._threadPool.start(worker)
