from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import time
import pickle

from libs.threading import *
from libs.objectDetector import *
from libs.trainingData import *
from libs.trainingSettings import *
from libs.namedImage import *

ADD_MODEL_COMMAND = '<< Add Model >>'
RUN_DETECTION = 'RunDetection'
MODEL_LIST = 'ModelList'
CURRENT_MODEL_NAME = 'CurrentModelName'
SOURCE_MODEL_FOLDER = 'SourceModelFolder'
TRAIN_MODEL_FOLDER = 'TrainModelFolder'
INFERENCE_GRAPH_FOLDER = 'InferenceGraphFolder'


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

        self._detectionQueue = []
        self._detectionInProgress = False

        self._runDetection = settings.get(RUN_DETECTION, False)
        self._modelList = settings.get(MODEL_LIST, [])
        self._currentModelName = settings.get(CURRENT_MODEL_NAME, '')
        self._source_model_folder = settings.get(SOURCE_MODEL_FOLDER)
        self._train_model_folder = settings.get(TRAIN_MODEL_FOLDER)
        self._inference_graph_folder = settings.get(INFERENCE_GRAPH_FOLDER)

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

        self._trainingPropertiesButton = QPushButton("Set training properties")
        self._trainingPropertiesButton.clicked.connect(self._set_training_properties)

        self._exporting_data_in_progress = False
        self._cancel_exporting = False
        self._training_data = TrainingData()

        self._exportTrainingDataButton = QPushButton("Export training data")
        self._exportTrainingDataButton.clicked.connect(self._export_training_data)

        self._runTrainingButton = QPushButton("Train Model")
        self._runTrainingButton.clicked.connect(self._run_training)

        self._exportInferenceGraphButton = QPushButton("Export graph")
        self._exportInferenceGraphButton.clicked.connect(self._export_inference_graph)

        detectionLayout = QHBoxLayout()
        detectionLayout.addWidget(self._runDetectionCheckbox)
        detectionLayout.addWidget(self._detectionModelsCombobox)
        detectionLayout.addStretch()
        detectionLayout.addWidget(self._trainingPropertiesButton)
        detectionLayout.addWidget(self._exportTrainingDataButton)
        detectionLayout.addWidget(self._runTrainingButton)
        detectionLayout.addWidget(self._exportInferenceGraphButton)

        detectionContainer = QWidget()
        detectionContainer.setLayout(detectionLayout)
        self.setWidget(detectionContainer)

        if self._detectionModelsCombobox.currentText() != ADD_MODEL_COMMAND:
            self._currentModelName = self._detectionModelsCombobox.currentText()

    # Public methods
    def Settings(self):
        settings = {RUN_DETECTION: self._runDetection, MODEL_LIST: self._modelList,
                    CURRENT_MODEL_NAME: self._currentModelName,
                    SOURCE_MODEL_FOLDER: self._source_model_folder,
                    TRAIN_MODEL_FOLDER: self._train_model_folder,
                    INFERENCE_GRAPH_FOLDER: self._inference_graph_folder}

        return pickle.dumps(settings)

    def ProcessImage(self, image):
        self._detectionQueue.append(image)
        self._process_detection_queue()

    # signals
    objects_detected = pyqtSignal(tuple)

    # private methods
    def _set_training_properties(self):
        try:
            ts = TrainingSettings(self, self._source_model_folder, self._train_model_folder,
                                  self._inference_graph_folder)
            res = ts.exec_()
            if res:
                self._source_model_folder, self._train_model_folder, self._inference_graph_folder = ts.getResult()
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))

    def _export_training_data(self):
        if self._exporting_data_in_progress:
            self._cancel_exporting = True
            return

        self._exporting_data_in_progress = True
        self._cancel_exporting = False
        try:
            ps = ProgramState.getInstance()
            label_map_path = ps.labelMapPath
            label_save_dir = ps.defaultSaveDir
            source_model_folder = self._source_model_folder
            train_model_folder = self._train_model_folder

            worker = ProgressingWorker(self._export_training_data_func, label_map_path, label_save_dir,
                                       source_model_folder,
                                       train_model_folder)  # Any other args, kwargs are passed to the run function
            worker.signals.error.connect(self._on_export_training_data_error)
            worker.signals.finished.connect(self._on_export_training_data_finished)
            worker.signals.progress.connect(self._on_export_training_data_progress)

            # Execute
            self._threadPool.start(worker)
            self._exportTrainingDataButton.setText("Cancel export")
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))

    def _export_training_data_func(self, label_map_path: str, label_save_dir: str, source_model_folder: str,
                                   train_model_folder: str, progress_callback):
        res = self._training_data.exportData(label_map_path, label_save_dir, source_model_folder, train_model_folder,
                                             progress_callback)
        if not res:
            pass

    def _on_export_training_data_error(self, restuple):
        str = restuple[1]

    def _on_export_training_data_progress(self, step: int, status: str):
        s = step
        msg = status

        if self._cancel_exporting:
            self._training_data.cancel()

    def _on_export_training_data_finished(self):
        self._exportTrainingDataButton.setText("Export training data")
        self._exporting_data_in_progress = False
        self._cancel_exporting = False

    def _run_training(self):
        pass

    def _export_inference_graph(self):
        pass

    def _process_detection_queue(self):
        if self._detectionInProgress or self._currentObjectDetector is None:
            return

        if len(self._detectionQueue) != 0:
            self._detectionInProgress = True
            image = self._detectionQueue[0]
            self._detectionQueue = self._detectionQueue[1:]

            #  run detection in separate thread
            worker = Worker(self._detect_objects_func, image)  # Any other args, kwargs are passed to the run function
            worker.signals.result.connect(self._on_detection_result)
            worker.signals.error.connect(self._on_detection_error)
            worker.signals.finished.connect(self._on_detection_finished)

            # Execute
            self._threadPool.start(worker)

    def _detect_objects_func(self, image):
        self._detectionModelsCombobox.setEnabled(False)
        self._runDetectionCheckbox.setEnabled(False)
        return image.path, self._currentObjectDetector.detect(image)

    def _on_detection_result(self, detection_result):
        self.objects_detected.emit(detection_result)

    def _on_detection_error(self, restuple):
        str = restuple[2]

    def _on_detection_finished(self):
        self._detectionModelsCombobox.setEnabled(True)
        self._runDetectionCheckbox.setEnabled(True)
        self._detectionInProgress = False
        self._process_detection_queue()

    def _run_detection_changed(self, item=None):
        self._runDetection = self._runDetectionCheckbox.isChecked()
        self._detectionModelsCombobox.setVisible(self._runDetection)

    def _load_model_func(self, modelName):
        od = ObjectDetector(modelName)
        return od

    def _on_loading_result(self, objectDetector):
        if objectDetector != None:
            self._objectDetectors[self._currentModelName] = objectDetector
            self._currentObjectDetector = objectDetector
            # check if label maps are equivalent
            path_to_labels = ProgramState.getInstance().labelMapPath
            detector_label_map = objectDetector.getLabelMap()
            if not detector_label_map.IsEqual(LabelMap(path_to_labels)):
                emsg = QMessageBox(self)
                emsg.setIcon(QMessageBox.Warning)
                emsg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                emsg.setText('Inconcistent Label Maps')
                emsg.setInformativeText(
                    "Application's Label Map is not equal to Object Detector's Label Map. Do you want to change application Label Map?")
                emsg.setWindowTitle("ObjectDetector")
                response = emsg.exec_()

    def _on_loading_finished(self):
        self._detectionModelsCombobox.setEnabled(True)
        self._runDetectionCheckbox.setEnabled(True)
        self._process_detection_queue()

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

            # load model in separate thread
            worker = Worker(self._load_model_func,
                            newModelName)  # Any other args, kwargs are passed to the run function
            worker.signals.result.connect(self._on_loading_result)
            worker.signals.error.connect(self._on_loading_error)
            worker.signals.finished.connect(self._on_loading_finished)

            # Execute
            self._threadPool.start(worker)
