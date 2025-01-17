#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import distutils.spawn
import os.path
import platform
import re
import sys
import subprocess

from functools import partial
from collections import defaultdict

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip
        sip.setapi('QVariant', 2)

from libs.resources import *
from libs.constants import *
from libs.utils import *
from libs.settings import Settings
from libs.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR
from libs.stringBundle import StringBundle
from libs.canvas import Canvas
from libs.zoomWidget import ZoomWidget
from libs.labelDialog import LabelDialog
from libs.colorDialog import ColorDialog
from libs.labelFile import LabelFile, LabelFileError
from libs.toolBar import ToolBar
from libs.pascal_voc_io import PascalVocReader
from libs.pascal_voc_io import XML_EXT
from libs.yolo_io import YoloReader
from libs.yolo_io import TXT_EXT
from libs.ustr import ustr
from libs.hashableQListWidgetItem import HashableQListWidgetItem
from libs.recognitionWidget import *
from libs.detectedShape import *
from libs.labelMap import *
from libs.videoFile import *
from libs.imagesList import *
from libs.folderImagesSource import *
from libs.videoImagesSource import *

__appname__ = 'LabelVideo'

class WindowMixin(object):

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(u'%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            addActions(toolbar, actions)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        return toolbar


class MainWindow(QMainWindow, WindowMixin):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self, defaultFilename=None, defaultPredefClassFile=None, defaultSaveDir=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        # Load setting in the main thread
        self.settings = Settings()
        self.settings.load()
        settings = self.settings

        # Load string bundle for i18n
        self.stringBundle = StringBundle.getBundle()
        getStr = lambda strId: self.stringBundle.getString(strId)

        # Save as Pascal voc xml
        program_state = ProgramState.getInstance()
        program_state.defaultSaveDir = defaultSaveDir

        self.usingPascalVocFormat = True
        self.usingYoloFormat = False

        # For loading all image under a directory
        program_state.labelMapPath = settings.get(SETTING_LABELMAP_FILENAME, defaultPredefClassFile)
        self.labelMap = LabelMap(program_state.labelMapPath)
        program_state.signals.labelMapPathChanged.connect(self.onLabelMapPathChanged)

        self._last_open_dir = None
        self._current_image = NamedImage("")

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False
        self._beginner = True
        self.screencastViewer = self.getAvailableScreencastViewer()
        self.screencast = "https://youtu.be/p0nR2YsCY_U"

        # Load predefined classes to the list

        # Main widgets and related state.
        self.labelDialog = LabelDialog(parent=self, listItem=self.labelMap.getLabels())

        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.itemsToDetectedShapes = {}
        self.detectedShapesToItems = {}
        self.prevLabelText = ''

        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)

        l1 = QLabel()
        l1.setText('Labels')
        listLayout.addWidget(l1)

        # Create and add a widget for showing current label items
        self.labelList = QListWidget()
        self.labelList.itemActivated.connect(self.labelSelectionChanged)
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self.editLabel)

        # Connect to itemChanged to detect checkbox changes.
        self.labelList.itemChanged.connect(self.labelItemChanged)
        listLayout.addWidget(self.labelList)

        l2 = QLabel()
        l2.setText('Detected:')

        convert_all_button = QPushButton('All')
        width = convert_all_button.fontMetrics().boundingRect("All").width() + 12
        convert_all_button.setMaximumWidth(width)
        convert_all_button.clicked.connect(partial(self._create_shape_from_detected_shape, label = None,
                                                   enquireType = False, allLabels = True))

        convert_all_but_current_button = QPushButton('All but current')
        width = convert_all_but_current_button.fontMetrics().boundingRect("All but current").width() + 12
        convert_all_but_current_button.setMaximumWidth(width)
        convert_all_but_current_button.clicked.connect(partial(self._create_shape_from_detected_shape, label = None,
                                                       enquireType = True, allLabels = True))


        detected_labels_layout = QHBoxLayout()
        detected_labels_layout.setContentsMargins(0, 0, 10, 0)
        detected_labels_layout.addWidget(l2)
        detected_labels_layout.addWidget(convert_all_button)
        detected_labels_layout.addWidget(convert_all_but_current_button)
        detected_labels_widget = QWidget()
        detected_labels_widget.setLayout(detected_labels_layout)


        listLayout.addWidget(detected_labels_widget)

        self.detectedLabelList = QListWidget()

        self.detectedLabelList.itemSelectionChanged.connect(self.detectedLabelSelectionChanged)
        self.detectedLabelList.itemActivated.connect(self.detectedLabelSelectionChanged)
        self.detectedLabelList.itemDoubleClicked.connect(partial(self._create_shape_from_detected_shape,
                                                                 label = None, enquireType = False, allLabels = False))
        self.detectedLabelList.itemChanged.connect(self.detectedLabelItemChanged)

        listLayout.addWidget(self.detectedLabelList)

        labelListContainer = QWidget()
        labelListContainer.setLayout(listLayout)

        self.dock = QDockWidget(getStr('boxLabelText'), self)
        self.dock.setObjectName(getStr('labels'))
        self.dock.setWidget(labelListContainer)

        # create images list widget
        self.imagesListDock = ImagesList(getStr('imagesList'), self.settings.get(SETTING_IMAGES_LIST), self)
        self.imagesListDock.image_changed.connect(self.onImageChanged)

        # create auto-recognition properties widget
        self.recognitionDock = Recognition(getStr('recognitionProperties'), self.settings.get(SETTING_AUTO_DETECTION), self)
        self.recognitionDock.objects_detected.connect(self.onObjectsDetected)

        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=self)

        self.canvas = Canvas(parent=self)
        self.canvas.zoomRequest.connect(self.zoomRequest)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scrollArea = scroll
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)

        self.dockFeatures = QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable
        self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

        # Actions
        action = partial(newAction, self)
        quit = action(getStr('quit'), self.close,
                      'Ctrl+Q', 'quit', getStr('quitApp'))

        openLabelMap = action(getStr('openLabelFile'), self.openLabelMapFile,
                      'Ctrl+O', 'open', getStr('openLabelFileDetail'))

        openVideoFile = action('Load video', self.loadVideoFile,
                      'Ctrl+Shift+V', 'open', 'Load video file for to label frames')

        opendir = action(getStr('openDir'), self.openDirDialog,
                         'Ctrl+u', 'open', getStr('openDir'))

        changeSavedir = action(getStr('changeSaveDir'), self.changeSavedirDialog,
                               'Ctrl+r', 'open', getStr('changeSavedAnnotationDir'))

        openNextImg = action(getStr('nextImg'), self.openNextImg,
                             'd', 'next', getStr('nextImgDetail'))

        openPrevImg = action(getStr('prevImg'), self.openPrevImg,
                             'a', 'prev', getStr('prevImgDetail'))

        verify = action(getStr('verifyImg'), self.verifyImg,
                        'space', 'verify', getStr('verifyImgDetail'))

        save = action(getStr('save'), self.saveFile,
                      'Ctrl+S', 'save', getStr('saveDetail'), enabled=False)

        save_format = action('&PascalVOC', self.change_format,
                      'Ctrl+', 'format_voc', getStr('changeSaveFormat'), enabled=True)

        saveAs = action(getStr('saveAs'), self.saveFileAs,
                        'Ctrl+Shift+S', 'save-as', getStr('saveAsDetail'), enabled=False)

        close = action(getStr('closeCur'), self.closeFile, 'Ctrl+W', 'close', getStr('closeCurDetail'))

        resetAll = action(getStr('resetAll'), self.resetAll, None, 'resetall', getStr('resetAllDetail'))

        color1 = action(getStr('boxLineColor'), self.chooseColor1,
                        'Ctrl+Shift+L', 'color_line', getStr('boxLineColorDetail'))

        createMode = action(getStr('crtBox'), self.setCreateMode,
                            'w', 'new', getStr('crtBoxDetail'), enabled=False)
        editMode = action('&Edit RectBox', self.setEditMode,
                          'Ctrl+J', 'edit', u'Move and edit Boxs', enabled=False)

        create = action(getStr('crtBox'), self.createShape,
                        'w', 'new', getStr('crtBoxDetail'), enabled=False)
        delete = action(getStr('delBox'), self.deleteSelectedShape,
                        'Delete', 'delete', getStr('delBoxDetail'), enabled=False)
        copy = action(getStr('dupBox'), self.copySelectedShape,
                      'Ctrl+D', 'copy', getStr('dupBoxDetail'),
                      enabled=False)

        advancedMode = action(getStr('advancedMode'), self.toggleAdvancedMode,
                              'Ctrl+Shift+A', 'expert', getStr('advancedModeDetail'),
                              checkable=True)

        hideAll = action('&Hide RectBox', partial(self.togglePolygons, False),
                         'Ctrl+H', 'hide', getStr('hideAllBoxDetail'),
                         enabled=False)
        showAll = action('&Show RectBox', partial(self.togglePolygons, True),
                         'Ctrl+A', 'hide', getStr('showAllBoxDetail'),
                         enabled=False)

        help = action(getStr('tutorial'), self.showTutorialDialog, None, 'help', getStr('tutorialDetail'))
        showInfo = action(getStr('info'), self.showInfoDialog, None, 'help', getStr('info'))

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        zoomIn = action(getStr('zoomin'), partial(self.addZoom, 10),
                        'Ctrl++', 'zoom-in', getStr('zoominDetail'), enabled=False)
        zoomOut = action(getStr('zoomout'), partial(self.addZoom, -10),
                         'Ctrl+-', 'zoom-out', getStr('zoomoutDetail'), enabled=False)
        zoomOrg = action(getStr('originalsize'), partial(self.setZoom, 100),
                         'Ctrl+=', 'zoom', getStr('originalsizeDetail'), enabled=False)
        fitWindow = action(getStr('fitWin'), self.setFitWindow,
                           'Ctrl+F', 'fit-window', getStr('fitWinDetail'),
                           checkable=True, enabled=False)
        fitWidth = action(getStr('fitWidth'), self.setFitWidth,
                          'Ctrl+Shift+F', 'fit-width', getStr('fitWidthDetail'),
                          checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut,
                       zoomOrg, fitWindow, fitWidth)
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(getStr('editLabel'), self.editLabel, 'Ctrl+E', 'edit', getStr('editLabelDetail'), enabled=True)

        shapeLineColor = action(getStr('shapeLineColor'), self.chshapeLineColor,
                                icon='color_line', tip=getStr('shapeLineColorDetail'),
                                enabled=False)
        shapeFillColor = action(getStr('shapeFillColor'), self.chshapeFillColor,
                                icon='color', tip=getStr('shapeFillColorDetail'),
                                enabled=False)

        create_same_label = action('Convert to same label',
                                   partial(self._create_shape_from_detected_shape, label = None,
                                           enquireType = False, allLabels = False), 'Ctrl+L', 'create',
                                   'Create label of the same type', enabled=True)

        create_all_same_labels = action('Convert all detected labels',
                                        partial(self._create_shape_from_detected_shape, label = None,
                                                enquireType = False, allLabels = True), '', 'create',
                                        'Create all labels of the same type', enabled=True)

        create_all_labels_but_current = action('Convert all but current detected labels',
                                        partial(self._create_shape_from_detected_shape, label = None,
                                                enquireType = True, allLabels = True), '', 'create',
                                        'Create all labels of the same type', enabled=True)

        create_garbage_label = action('Convert to garbage label',
                                      partial(self._create_shape_from_detected_shape, label ='garbage',
                                              enquireType = False, allLabels = False), 'Ctrl+G', 'create',
                                      'Create label of the garbage_bag type', enabled=True)

        create_label_of_type = action('Convert to label of type...',
                                      partial(self._create_shape_from_detected_shape, label = None,
                                              enquireType = True, allLabels = False), 'Ctrl+T', 'create',
                                     'Specify type for new label', enabled=True)


        labels = self.dock.toggleViewAction()
        labels.setText(getStr('showHide'))
        labels.setShortcut('Ctrl+Shift+H')

        # Lavel list context menu.
        labelMenu = QMenu()
        addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(self.popLabelListMenu)

        detectedLabelMenu = QMenu()
        addActions(detectedLabelMenu, (create_same_label, create_garbage_label, create_label_of_type,
                                       create_all_same_labels, create_all_labels_but_current))
        self.detectedLabelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.detectedLabelList.customContextMenuRequested.connect(self.popDetectedLabelListMenu)

        # Store actions for further handling.
        self.actions = struct(save=save, save_format=save_format, saveAs=saveAs, open=openLabelMap, close=close,
                              resetAll = resetAll, lineColor=color1, create=create, delete=delete,
                              edit=edit, copy=copy, createMode=createMode, editMode=editMode,
                              advancedMode=advancedMode, shapeLineColor=shapeLineColor,
                              shapeFillColor=shapeFillColor,
                              zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
                              fitWindow=fitWindow, fitWidth=fitWidth,
                              zoomActions=zoomActions,
                              fileMenuActions=(openLabelMap, openVideoFile, opendir, save, saveAs, close, resetAll, quit),
                              beginner=(),
                              advanced=(),
                              editMenu=(edit, copy, delete, None, color1),
                              beginnerContext=(create, edit, copy, delete),
                              advancedContext=(createMode, editMode, edit, copy, delete, shapeLineColor, shapeFillColor),
                              onLoadActive=(close, create, createMode, editMode),
                              onShapesPresent=(saveAs, hideAll, showAll))

        self.menus = struct(
            file=self.menu('&File'),
            edit=self.menu('&Edit'),
            view=self.menu('&View'),
            help=self.menu('&Help'),
            recentFiles=QMenu('Open &Recent'),
            labelList=labelMenu,
            detectedLabelList=detectedLabelMenu)

        # Auto saving : Enable auto saving if pressing next
        self.autoSaving = QAction(getStr('autoSaveMode'), self)
        self.autoSaving.setCheckable(True)
        self.autoSaving.setChecked(settings.get(SETTING_AUTO_SAVE, False))
        # Sync single class mode from PR#106
        self.singleClassMode = QAction(getStr('singleClsMode'), self)
        self.singleClassMode.setShortcut("Ctrl+Shift+S")
        self.singleClassMode.setCheckable(True)
        self.singleClassMode.setChecked(settings.get(SETTING_SINGLE_CLASS, False))
        self.lastLabel = None
        # Add option to enable/disable labels being displayed at the top of bounding boxes
        self.displayLabelOption = QAction(getStr('displayLabel'), self)
        self.displayLabelOption.setShortcut("Ctrl+Shift+P")
        self.displayLabelOption.setCheckable(True)
        self.displayLabelOption.setChecked(settings.get(SETTING_PAINT_LABEL, False))
        self.displayLabelOption.triggered.connect(self.togglePaintLabelsOption)
        # Auto Dir/File restore on start
        self.autoRestore = QAction('Restore folder/file on start' , self)
        self.autoRestore.setShortcut("Ctrl+Shift+R")
        self.autoRestore.setCheckable(True)
        self.autoRestore.setChecked(settings.get(SETTING_RESTORE_ON_START, True))

        addActions(self.menus.file,
                   (openLabelMap, openVideoFile, opendir, changeSavedir, self.autoRestore, self.menus.recentFiles, save, save_format, saveAs, close, resetAll, quit))

        addActions(self.menus.help, (help, showInfo))
        addActions(self.menus.view, (
            self.autoSaving,
            self.singleClassMode,
            self.displayLabelOption,
            labels, advancedMode, None,
            hideAll, showAll, None,
            zoomIn, zoomOut, zoomOrg, None,
            fitWindow, fitWidth))

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        addActions(self.canvas.menus[0], self.actions.beginnerContext)
        addActions(self.canvas.menus[1], (
            action('&Copy here', self.copyShape),
            action('&Move here', self.moveShape)))

        self.tools = self.toolbar('Tools')
        self.actions.beginner = (
            openLabelMap, openVideoFile, opendir, changeSavedir, openNextImg, openPrevImg, verify, save, save_format, None, create, copy, delete, None,
            zoomIn, zoom, zoomOut, fitWindow, fitWidth)

        self.actions.advanced = (
            openLabelMap, openVideoFile, opendir, changeSavedir, openNextImg, openPrevImg, save, save_format, None,
            createMode, editMode, None,
            hideAll, showAll)

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        self._file_path = None
        self._video_file_path = None
        if defaultFilename is not None:
            if os.path.splitext(defaultFilename) in ('mov', 'avi'):
                self._video_file_path = defaultFilename
            else:
                self._file_path = ustr(defaultFilename)

        self.recentFiles = []
        self.maxRecent = 7
        self.lineColor = None
        self.fillColor = None
        self.zoom_level = 100
        self.fit_window = False
        # Add Chris
        self.difficult = False

        ## Fix the compatible issue for qt4 and qt5. Convert the QStringList to python list
        if settings.get(SETTING_RECENT_FILES):
            if have_qstring():
                recentFileQStringList = settings.get(SETTING_RECENT_FILES)
                self.recentFiles = [ustr(i) for i in recentFileQStringList]
            else:
                self.recentFiles = recentFileQStringList = settings.get(SETTING_RECENT_FILES)

        size = settings.get(SETTING_WIN_SIZE, QSize(600, 500))
        position = QPoint(0, 0)
        saved_position = settings.get(SETTING_WIN_POSE, position)
        # Fix the multiple monitors issue
        for i in range(QApplication.desktop().screenCount()):
            if QApplication.desktop().availableGeometry(i).contains(saved_position):
                position = saved_position
                break
        self.resize(size)
        self.move(position)
        saveDir = ustr(settings.get(SETTING_SAVE_DIR, None))

        self._storage_type = settings.get(SETTING_STORAGE_TYPE, STORAGE_TYPE_FOLDER)
        self._last_open_dir = ustr(settings.get(SETTING_LAST_OPEN_DIR, None))
        self._file_path = ustr(settings.get(SETTING_FILENAME, None))
        self._video_file_path = ustr(settings.get(SETTING_VIDEO_FILENAME, None))

        if program_state.defaultSaveDir is None and saveDir is not None and os.path.exists(saveDir):
            program_state.defaultSaveDir = saveDir
            self.statusBar().showMessage('%s started. Annotation will be saved to %s' %
                                         (__appname__, program_state.defaultSaveDir))
            self.statusBar().show()

        self.restoreState(settings.get(SETTING_WIN_STATE, QByteArray()))
        Shape.line_color = self.lineColor = QColor(settings.get(SETTING_LINE_COLOR, DEFAULT_LINE_COLOR))
        Shape.fill_color = self.fillColor = QColor(settings.get(SETTING_FILL_COLOR, DEFAULT_FILL_COLOR))
        self.canvas.setDrawingColor(self.lineColor)
        # Add chris
        Shape.difficult = self.difficult

        def xbool(x):
            if isinstance(x, QVariant):
                return x.toBool()
            return bool(x)

        if xbool(settings.get(SETTING_ADVANCE_MODE, False)):
            self.actions.advancedMode.setChecked(True)
            self.toggleAdvancedMode()

        # Populate the File menu dynamically.
        self.updateFileMenu()

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # Display cursor coordinates at the right of status bar
        self.labelCoordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.labelCoordinates)

        if self.autoRestore.isChecked():
            if self._storage_type == STORAGE_TYPE_VIDEO and self._video_file_path and os.path.isfile(self._video_file_path):
                self.queueEvent(partial(self.setSourceVideo, ustr(self._video_file_path), self._file_path))
            elif self._storage_type == STORAGE_TYPE_FOLDER and self._last_open_dir and os.path.isdir(self._last_open_dir):
                targetDirPath = ustr(self._last_open_dir)
                self.queueEvent(partial(self.setSourceFolder, targetDirPath, self._file_path))


    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.canvas.setDrawingShapeToSquare(False)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            # Draw rectangle if Ctrl is pressed
            self.canvas.setDrawingShapeToSquare(True)

    ## Support Functions ##
    def set_format(self, save_format):
        if save_format == FORMAT_PASCALVOC:
            self.actions.save_format.setText(FORMAT_PASCALVOC)
            self.actions.save_format.setIcon(newIcon("format_voc"))
            self.usingPascalVocFormat = True
            self.usingYoloFormat = False
            LabelFile.suffix = XML_EXT

        elif save_format == FORMAT_YOLO:
            self.actions.save_format.setText(FORMAT_YOLO)
            self.actions.save_format.setIcon(newIcon("format_yolo"))
            self.usingPascalVocFormat = False
            self.usingYoloFormat = True
            LabelFile.suffix = TXT_EXT

    def change_format(self):
        if self.usingPascalVocFormat: self.set_format(FORMAT_YOLO)
        elif self.usingYoloFormat: self.set_format(FORMAT_PASCALVOC)

    def noShapes(self):
        return not self.itemsToShapes

    def toggleAdvancedMode(self, value=True):
        self._beginner = not value
        self.canvas.setEditing(True)
        self.populateModeActions()
        if value:
            self.actions.createMode.setEnabled(True)
            self.actions.editMode.setEnabled(False)
            self.dock.setFeatures(self.dock.features() | self.dockFeatures)
        else:
            self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

    def populateModeActions(self):
        if self.beginner():
            tool, menu = self.actions.beginner, self.actions.beginnerContext
        else:
            tool, menu = self.actions.advanced, self.actions.advancedContext
        self.tools.clear()
        addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (self.actions.create,) if self.beginner()\
            else (self.actions.createMode, self.actions.editMode)
        addActions(self.menus.edit, actions + self.actions.editMenu)

    def setBeginner(self):
        self.tools.clear()
        addActions(self.tools, self.actions.beginner)

    def setAdvanced(self):
        self.tools.clear()
        addActions(self.tools, self.actions.advanced)

    def setDirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.create.setEnabled(True)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        self.labelList.clear()
        self.itemsToDetectedShapes.clear()
        self.detectedShapesToItems.clear()
        self.detectedLabelList.clear()
        self._file_path = None
        self.labelFile = None
        self.canvas.resetState()
        self.labelCoordinates.clear()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def currentDetectedItem(self):
        items = self.detectedLabelList.selectedItems()
        if items:
            return items[0]

        return None

    def addRecentFile(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filePath)

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()

    def getAvailableScreencastViewer(self):
        osName = platform.system()

        if osName == 'Windows':
            return ['C:\\Program Files\\Internet Explorer\\iexplore.exe']
        elif osName == 'Linux':
            return ['xdg-open']
        elif osName == 'Darwin':
            return ['open']

    ## Callbacks ##
    def onObjectsDetected(self, detection_result):
        if detection_result[0] == self._file_path:
            self.canvas.loadDetectedShapes(detection_result[1])
            for detectedShape in detection_result[1]:
                self.addDetectedLabel(detectedShape)

    def onImageChanged(self, image):
        self.resetState()

        self._current_image = image
        self.labelFile = None
        self.canvas.verified = False

        self.status("Loaded %s" % os.path.basename(image.path))
        self._file_path = image.path
        try:
            self.canvas.loadPixmap(QPixmap.fromImage(self._current_image.qtimage))

            self.setClean()
            self.canvas.setEnabled(True)
            self.adjustScale(initial=True)
            self.paintCanvas()
            self.addRecentFile(self._file_path)
            self.toggleActions(True)

            # start object detection
            self.recognitionDock.ProcessImage(image)

            # Label xml file and show bound box according to its filename
            # if self.usingPascalVocFormat is True:
            program_state = ProgramState.getInstance()
            if program_state.defaultSaveDir is not None:
                basename = os.path.basename(os.path.splitext(image.savepath)[0])
                xmlPath = os.path.join(program_state.defaultSaveDir, basename + XML_EXT)

                if os.path.isfile(xmlPath):
                    self.loadPascalXMLByFilename(xmlPath)

            self.setWindowTitle(__appname__ + ' ' + self._file_path)

            # Default : select last item if there is at least one item
            if self.labelList.count():
                self.labelList.setCurrentItem(self.labelList.item(self.labelList.count() - 1))
                self.labelList.item(self.labelList.count() - 1).setSelected(True)

            self.canvas.setFocus(True)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            err = traceback.format_exc()

        return True

    def showTutorialDialog(self):
        subprocess.Popen(self.screencastViewer + [self.screencast])

    def showInfoDialog(self):
        msg = u'Name:{0} \nApp Version:{1} \n{2} '.format(__appname__, __version__, sys.version_info)
        QMessageBox.information(self, u'Information', msg)

    def createShape(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.actions.create.setEnabled(False)

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)
        if not drawing and self.beginner():
            # Cancel creation.
            print('Cancel creation.')
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()
            self.actions.create.setEnabled(True)

    def toggleDrawMode(self, edit=True):
        self.canvas.setEditing(edit)
        self.actions.createMode.setEnabled(edit)
        self.actions.editMode.setEnabled(not edit)

    def setCreateMode(self):
        assert self.advanced()
        self.toggleDrawMode(False)

    def setEditMode(self):
        assert self.advanced()
        self.toggleDrawMode(True)
        self.labelSelectionChanged()

    def updateFileMenu(self):
        currFilePath = self._file_path

        def exists(filename):
            return os.path.exists(filename)
        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f !=
                 currFilePath and exists(f)]
        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def popDetectedLabelListMenu(self, point):
        self.menus.detectedLabelList.exec_(self.detectedLabelList.mapToGlobal(point))

    def editLabel(self):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:
            return
        text = self.labelDialog.popUp(item.text())
        if text is not None:
            item.setText(text)
            item.setBackground(generateColorByText(text))
            self.setDirty()

    # Add chris
    def btnstate(self, item= None):
        """ Function to handle difficult examples
        Update on each object """
        if not self.canvas.editing():
            return

        item = self.currentItem()
        if not item: # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count()-1)

        try:
            shape = self.itemsToShapes[item]
        except:
            pass
        # Checked and Update
        try:
            if difficult != shape.difficult:
                shape.difficult = difficult
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass

    # React to canvas signals.
    def shapeSelectionChanged(self, selected=False):
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:
            shape = self.canvas.selectedShape
            if shape:
                self.shapesToItems[shape].setSelected(True)
            else:
                self.labelList.clearSelection()
        self.actions.delete.setEnabled(selected)
        self.actions.copy.setEnabled(selected)
        self.actions.edit.setEnabled(selected)
        self.actions.shapeLineColor.setEnabled(selected)
        self.actions.shapeFillColor.setEnabled(selected)

    def addLabel(self, shape):
        shape.paintLabel = self.displayLabelOption.isChecked()
        item = HashableQListWidgetItem(shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        item.setBackground(generateColorByText(shape.label))
        self.itemsToShapes[item] = shape
        self.shapesToItems[shape] = item
        self.labelList.addItem(item)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

    def addDetectedLabel(self, detectedShape):
        dlbl = str.format('{0} -- {1}%', detectedShape.label, int(detectedShape.score * 100))
        item = HashableQListWidgetItem(dlbl)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)

        for lbl in self.itemsToShapes.keys():
            txt = lbl.text()
            if txt == detectedShape.label or txt == 'garbage':
                shp = self.itemsToShapes[lbl]
                xtnt = shp.getExtent()
                dxtnt = detectedShape.extent
                if self.checkExtents(xtnt, dxtnt):
                    item.setCheckState(Qt.Unchecked)
                    detectedShape.visible = False
                    fnt = QFont()
                    fnt.setBold(True)
                    item.setFont(fnt)
                    if txt == 'garbage':
                        item.setText(dlbl + ' : garbaged')
                        detectedShape.garbaged = True

        self.itemsToDetectedShapes[item] = detectedShape
        self.detectedShapesToItems[detectedShape] = item
        self.detectedLabelList.addItem(item)
        self.canvas.repaint()

    def checkExtents(self, ex1, ex2):
        if not self.pointInside(ex1[0], ex1[1], ex2): return False
        if not self.pointInside(ex1[2], ex1[1], ex2): return False
        if not self.pointInside(ex1[2], ex1[3], ex2): return False
        if not self.pointInside(ex1[0], ex1[3], ex2): return False

        if not self.pointInside(ex2[0], ex2[1], ex1): return False
        if not self.pointInside(ex2[2], ex2[1], ex1): return False
        if not self.pointInside(ex2[2], ex2[3], ex1): return False
        if not self.pointInside(ex2[0], ex2[3], ex1): return False
        return True

    def pointInside(self, x, y, extent):
        xi = extent[0] - (extent[2] - extent[0]) * 0.1
        xa = extent[2] + (extent[2] - extent[0]) * 0.1
        if not xi < x < xa : return False

        yi = extent[1] - (extent[3] - extent[1]) * 0.1
        ya = extent[3] + (extent[3] - extent[1]) * 0.1
        if not yi < y < ya : return False

        return True

    def remLabel(self, shape):
        if shape is None:
            # print('rm empty label')
            return
        item = self.shapesToItems[shape]
        self.labelList.takeItem(self.labelList.row(item))
        del self.shapesToItems[shape]
        del self.itemsToShapes[item]

    def loadLabels(self, shapes):
        s = []
        for label, points, line_color, fill_color, difficult in shapes:
            shape = Shape(label=label)
            for x, y in points:

                # Ensure the labels are within the bounds of the image. If not, fix them.
                x, y, snapped = self.canvas.snapPointToCanvas(x, y)
                if snapped:
                    self.setDirty()

                shape.addPoint(QPointF(x, y))
            shape.difficult = difficult
            shape.close()
            s.append(shape)

            if line_color:
                shape.line_color = QColor(*line_color)
            else:
                shape.line_color = generateColorByText(label)

            if fill_color:
                shape.fill_color = QColor(*fill_color)
            else:
                shape.fill_color = generateColorByText(label)

            self.addLabel(shape)

        self.canvas.loadShapes(s)

    def saveLabels(self, annotationFilePath, imageFilePath):
        annotationFilePath = ustr(annotationFilePath)
        imageFilePath = ustr(imageFilePath)
        if self.labelFile is None:
            self.labelFile = LabelFile()
            self.labelFile.verified = self.canvas.verified

        def format_shape(s):
            return dict(label=s.label,
                        line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        points=[(p.x(), p.y()) for p in s.points],
                       # add chris
                        difficult = s.difficult)

        shapes = [format_shape(shape) for shape in self.canvas.shapes]
        # Can add differrent annotation formats here
        try:
            if self.usingPascalVocFormat is True:
                if annotationFilePath[-4:].lower() != ".xml":
                    annotationFilePath += XML_EXT
                self.labelFile.savePascalVocFormat(annotationFilePath, shapes, imageFilePath,
                                                   self.lineColor.getRgb(), self.fillColor.getRgb())
            elif self.usingYoloFormat is True:
                if annotationFilePath[-4:].lower() != ".txt":
                    annotationFilePath += TXT_EXT
                self.labelFile.saveYoloFormat(annotationFilePath, shapes, imageFilePath, self.labelMap.getLabels(),
                                              self.lineColor.getRgb(), self.fillColor.getRgb())
            else:
                self.labelFile.save(annotationFilePath, shapes, imageFilePath,
                                    self.lineColor.getRgb(), self.fillColor.getRgb())
            print('Image:{0} -> Annotation:{1}'.format(self._file_path, annotationFilePath))
            return True
        except LabelFileError as e:
            self.errorMessage(u'Error saving label data', u'<b>%s</b>' % e)
            return False

    def copySelectedShape(self):
        self.addLabel(self.canvas.copySelectedShape())
        # fix copy and delete
        self.shapeSelectionChanged(True)

    def detectedLabelSelectionChanged(self):
        item = self.currentDetectedItem()
        if item and self.canvas.editing():
            self._noSelectionSlot = True
            self.canvas.selectDetectedShape(self.itemsToDetectedShapes[item])

    def onLabelMapPathChanged(self, label_map_path):
        self.labelMap = LabelMap(label_map_path)
        self.labelDialog = LabelDialog(parent=self, listItem=self.labelMap.getLabels())

    def detectedLabelItemChanged(self, item):
        shape = self.itemsToDetectedShapes[item]
        self.canvas.setDetectedShapeVisible(shape, item.checkState() == Qt.Checked)

    def labelSelectionChanged(self):
        item = self.currentItem()
        if item and self.canvas.editing():
            self._noSelectionSlot = True
            self.canvas.selectShape(self.itemsToShapes[item])
            shape = self.itemsToShapes[item]

    def _create_shape_from_detected_shape(self, label = None, enquireType = False, allLabels = False):
        current_item = self.currentDetectedItem()
        items = {}
        if allLabels:
            # convert all labels
            if enquireType:
                # convert all but current
                if not current_item:
                    return

                for item in self.itemsToDetectedShapes.keys():
                    if item != current_item:
                        items[item] = self.itemsToDetectedShapes[item].label

            else:
                for item in self.itemsToDetectedShapes.keys():
                    items[item] = self.itemsToDetectedShapes[item].label


        else:
            # convert specific label
            if not current_item:
                return

            text = label
            if label is None:
                text = self.itemsToDetectedShapes[current_item].label

            if enquireType:
                text = self.labelDialog.popUp(text=self.prevLabelText)
                if text is None:
                    return

            items[current_item] = text

        try:
            for item in items.keys():
                shp = self.itemsToDetectedShapes[item]
                text = items[item]

                xmin, ymin, xmax, ymax = shp.extent
                points = [QPointF(xmin, ymin), QPointF(xmax, ymin), QPointF(xmax, ymax), QPointF(xmin, ymax)]

                newshape = Shape(label=text, points=points)
                newshape.fill_color = generateColorByText(text)
                newshape.line_color = newshape.fill_color

                item.setCheckState(Qt.Unchecked)
                shp.visible = False
                fnt = QFont()
                fnt.setBold(True)
                fnt.setItalic(True)
                item.setFont(fnt)

                self.addLabel(newshape)
                self.canvas.addShape(newshape)
                self.setDirty()
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            fmt = traceback.format_exc()


    def labelItemChanged(self, item):
        shape = self.itemsToShapes[item]
        label = item.text()
        if label != shape.label:
            shape.label = item.text()
            shape.line_color = generateColorByText(shape.label)
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    # Callback functions:
    def newShape(self, newshape):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """

        if len(self.labelMap.getLabels()) > 0:
            self.labelDialog = LabelDialog(parent=self, listItem=self.labelMap.getLabels())

        # Sync single class mode from PR#106
        if self.singleClassMode.isChecked() and self.lastLabel:
            text = self.lastLabel
        else:
            text = self.labelDialog.popUp(text=self.prevLabelText)
            self.lastLabel = text

        if text is not None:
            self.prevLabelText = text

            generate_color = generateColorByText(text)
            newshape.label = text
            newshape.line_color = generate_color
            newshape.fill_color = generate_color

            self.addLabel(newshape)
            if self.beginner():  # Switch to edit mode.
                self.canvas.setEditing(True)
                self.actions.create.setEnabled(True)
        else:
            self.canvas.resetAllLines()

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    def zoomRequest(self, delta):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scrollArea.width()
        h = self.scrollArea.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # zoom in
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item, shape in self.itemsToShapes.items():
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def resizeEvent(self, event):
        if self.canvas and not self._current_image.isNull() and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()

        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self._current_image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        settings = self.settings

        settings[SETTING_LABELMAP_FILENAME] = ProgramState.getInstance().labelMapPath
        settings[SETTING_FILENAME] = self._file_path if self._file_path else ''
        settings[SETTING_WIN_SIZE] = self.size()
        settings[SETTING_WIN_POSE] = self.pos()
        settings[SETTING_WIN_STATE] = self.saveState()
        settings[SETTING_LINE_COLOR] = self.lineColor
        settings[SETTING_FILL_COLOR] = self.fillColor
        settings[SETTING_RECENT_FILES] = self.recentFiles
        settings[SETTING_ADVANCE_MODE] = not self._beginner
        program_state = ProgramState.getInstance()
        if program_state.defaultSaveDir and os.path.exists(program_state.defaultSaveDir):
            settings[SETTING_SAVE_DIR] = ustr(program_state.defaultSaveDir)
        else:
            settings[SETTING_SAVE_DIR] = ''

        if self._last_open_dir:
            settings[SETTING_LAST_OPEN_DIR] = self._last_open_dir
        else:
            settings[SETTING_LAST_OPEN_DIR] = ''

        if self._video_file_path:
            settings[SETTING_VIDEO_FILENAME] = self._video_file_path
        else:
            settings[SETTING_VIDEO_FILENAME] = ''

        settings[SETTING_STORAGE_TYPE] = self._storage_type

        settings[SETTING_AUTO_SAVE] = self.autoSaving.isChecked()
        settings[SETTING_SINGLE_CLASS] = self.singleClassMode.isChecked()
        settings[SETTING_PAINT_LABEL] = self.displayLabelOption.isChecked()
        settings[SETTING_RESTORE_ON_START] = self.autoRestore.isChecked()
        settings[SETTING_AUTO_DETECTION] = self.recognitionDock.Settings()
        settings.save()

    def loadRecent(self, filename):
        if self.mayContinue():
            self._load_file(filename)

    def changeSavedirDialog(self, _value=False):
        program_state = ProgramState.getInstance()
        if program_state.defaultSaveDir is not None:
            path = ustr(program_state.defaultSaveDir)
        else:
            path = '.'

        dirpath = ustr(QFileDialog.getExistingDirectory(self,
                                                       '%s - Save annotations to the directory' % __appname__, path,  QFileDialog.ShowDirsOnly
                                                       | QFileDialog.DontResolveSymlinks))

        if dirpath is not None and len(dirpath) > 1:
            program_state.defaultSaveDir = dirpath

        self.statusBar().showMessage('%s . Annotation will be saved to %s' %
                                     ('Change saved folder', program_state.defaultSaveDir))
        self.statusBar().show()

    def openDirDialog(self, _value=False, dirpath=None):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else '.'
        if self._last_open_dir and os.path.exists(self._last_open_dir):
            defaultOpenDirPath = self._last_open_dir
        else:
            defaultOpenDirPath = os.path.dirname(self._file_path) if self._file_path else '.'

        targetDirPath = ustr(QFileDialog.getExistingDirectory(self,
                                                     '%s - Open Directory' % __appname__, defaultOpenDirPath,
                                                     QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        if targetDirPath:
            self.setSourceFolder(targetDirPath, self._file_path)

    def setSourceFolder(self, dirpath, filename):
        if not self.mayContinue() or not dirpath:
            return

        self._storage_type = STORAGE_TYPE_FOLDER
        self._last_open_dir = dirpath
        images_source = FolderImagesSource(dirpath)
        self.imagesListDock.SetSource(images_source)
        self.imagesListDock.SetImage(filename)

    def setSourceVideo(self, videopath, filename):
        if not self.mayContinue() or not videopath:
            return

        self._storage_type = STORAGE_TYPE_VIDEO
        self._video_file_path = videopath
        images_source = VideoImagesSource(videopath)
        self.imagesListDock.SetSource(images_source)
        self.imagesListDock.SetImage(filename)

    def verifyImg(self, _value=False):
        # Proceding next image without dialog if having any label
        if self._file_path is not None:
            try:
                self.labelFile.toggleVerify()
            except AttributeError:
                # If the labelling file does not exist yet, create if and
                # re-save it with the verified attribute.
                self.saveFile()
                if self.labelFile != None:
                    self.labelFile.toggleVerify()
                else:
                    return

            self.canvas.verified = self.labelFile.verified
            self.paintCanvas()
            self.saveFile()

    def openPrevImg(self, _value=False):
        if self.autoSaving.isChecked():
            program_state = ProgramState.getInstance()
            if program_state.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedirDialog()
                return

        if not self.mayContinue():
            return

        self.imagesListDock.SetPrevImage()

    def openNextImg(self, _value=False):
        if self.autoSaving.isChecked():
            program_state = ProgramState.getInstance()
            if program_state.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedirDialog()
                return

        if not self.mayContinue():
            return

        self.imagesListDock.SetNextImage()

    def openLabelMapFile(self, _value=False):
        path = os.path.dirname(ustr(self._file_path)) if self._file_path else '.'
        filters = "TensorFlow Label Map files (*.pbtxt)"

        dlg = QFileDialog(self, '%s - Choose Label file' % __appname__, path, filters)
        dlg.setAcceptMode(QFileDialog.AcceptOpen)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        filename = None
        if dlg.exec_():
            filename = ustr(dlg.selectedFiles()[0])

        if filename:
            ProgramState.getInstance().labelMapPath = filename

    def loadVideoFile(self):
        path = os.path.dirname(ustr(self._video_file_path)) if self._video_file_path else '.'
        filters = "Video files (*.mov;*.avi;*mp4)"

        dlg = QFileDialog(self, '%s - Choose Video file' % __appname__, path, filters)
        dlg.setAcceptMode(QFileDialog.AcceptOpen)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)

        filename = None
        if dlg.exec_():
            filename = ustr(dlg.selectedFiles()[0])

        if filename:
            ProgramState.getInstance().videoFilePath = filename
            self.setSourceVideo(filename, "")

    def saveFile(self):
        imagepath = self._current_image.Save()

        program_state = ProgramState.getInstance()
        if program_state.defaultSaveDir is not None and len(ustr(program_state.defaultSaveDir)):
            if imagepath:
                imgFileName = os.path.basename(imagepath)
                savedFileName = os.path.splitext(imgFileName)[0]
                savedPath = os.path.join(ustr(program_state.defaultSaveDir), savedFileName)
                self.saveLabelsFile(savedPath, imagepath)
        else:
            imgFileDir = os.path.dirname(imagepath)
            imgFileName = os.path.basename(imagepath)
            savedFileName = os.path.splitext(imgFileName)[0]
            savedPath = os.path.join(imgFileDir, savedFileName)
            self._saveFile(savedPath if self.labelFile else self.saveFileDialog(removeExt=False))

    def saveFileAs(self):
        assert not self._current_image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self, removeExt=True):
        caption = '%s - Choose File' % __appname__
        filters = 'File (*%s)' % LabelFile.suffix
        openDialogPath = self.currentPath()
        dlg = QFileDialog(self, caption, openDialogPath, filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        filenameWithoutExtension = os.path.splitext(self._file_path)[0]
        dlg.selectFile(filenameWithoutExtension)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if dlg.exec_():
            fullFilePath = ustr(dlg.selectedFiles()[0])
            if removeExt:
                return os.path.splitext(fullFilePath)[0] # Return file path without the extension.
            else:
                return fullFilePath
        return ''

    def saveLabelsFile(self, labelsFilePath, imageFilePath):
        if labelsFilePath and self.saveLabels(labelsFilePath, imageFilePath):
            self.setClean()
            self.statusBar().showMessage('Saved to  %s' % labelsFilePath)
            self.statusBar().show()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def resetAll(self):
        self.settings.reset()
        self.close()
        proc = QProcess()
        proc.startDetached(os.path.abspath(__file__))

    def mayContinue(self):
        return not (self.dirty and not self.discardChangesDialog())

    def discardChangesDialog(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'You have unsaved changes, proceed anyway?'
        return yes == QMessageBox.warning(self, u'Attention', msg, yes | no)

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(self._file_path) if self._file_path else '.'

    def chooseColor1(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.lineColor = color
            Shape.line_color = color
            self.canvas.setDrawingColor(color)
            self.canvas.update()
            self.setDirty()

    def deleteSelectedShape(self):
        self.remLabel(self.canvas.deleteSelected())
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def chshapeLineColor(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.canvas.selectedShape.line_color = color
            self.canvas.update()
            self.setDirty()

    def chshapeFillColor(self):
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            self.canvas.selectedShape.fill_color = color
            self.canvas.update()
            self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addLabel(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def loadPascalXMLByFilename(self, xmlPath):
        if self._file_path is None:
            return
        if os.path.isfile(xmlPath) is False:
            return

        self.set_format(FORMAT_PASCALVOC)

        tVocParseReader = PascalVocReader(xmlPath)
        shapes = tVocParseReader.getShapes()
        self.loadLabels(shapes)
        self.canvas.verified = tVocParseReader.verified

    def loadYOLOTXTByFilename(self, txtPath):
        if self._file_path is None:
            return
        if os.path.isfile(txtPath) is False:
            return

        self.set_format(FORMAT_YOLO)
        tYoloParseReader = YoloReader(txtPath, self.image)
        shapes = tYoloParseReader.getShapes()
        print (shapes)
        self.loadLabels(shapes)
        self.canvas.verified = tYoloParseReader.verified

    def togglePaintLabelsOption(self):
        for shape in self.canvas.shapes:
            shape.paintLabel = self.displayLabelOption.isChecked()

def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])


def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    app = QApplication(argv)
    styles = QStyleFactory.keys()
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))
    app.setStyle('Fusion')

    # Tzutalin 201705+: Accept extra agruments to change predefined class file
    # Usage : LabelVideo.py image predefClasse saveDir
    win = MainWindow(argv[1] if len(argv) >= 2 else None,
                     argv[2] if len(argv) >= 3 else os.path.join(os.path.dirname(sys.argv[0]), 'data', 'labelmap.pbtxt'),
                     argv[3] if len(argv) >= 4 else None)
    win.show()
    return app, win


def main():
    '''construct main app and run it'''
    app, _win = get_main_app(sys.argv)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
