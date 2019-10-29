from PyQt5.QtCore import *

import traceback, sys

class ProgramStateSignals(QObject):
    labelMapPathChanged = pyqtSignal(str)
    videoFileChanged = pyqtSignal(str)
    defaultSaveDirChanged = pyqtSignal(str)

class ProgramState:
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if ProgramState.__instance == None:
            ProgramState()

        return ProgramState.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if ProgramState.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            ProgramState.__instance = self

        self.__labelMapPath = None
        self.__defaultSaveDir = None
        self.__videoFilePath = None

        self.signals = ProgramStateSignals()

    @property
    def labelMapPath(self):
        return self.__labelMapPath

    @labelMapPath.setter
    def labelMapPath(self, val):
        self.__labelMapPath = val
        self.signals.labelMapPathChanged.emit(self.__labelMapPath)

    @property
    def videoFilePath(self):
        return self.__videoFilePath

    @videoFilePath.setter
    def videoFilePath(self, val):
        self.__videoFilePath = val
        self.signals.videoFileChanged.emit(self.__videoFilePath)

    @property
    def defaultSaveDir(self):
        return self.__defaultSaveDir

    @defaultSaveDir.setter
    def defaultSaveDir(self, val):
        self.__defaultSaveDir = val
        self.signals.defaultSaveDirChanged.emit(self.__defaultSaveDir)

