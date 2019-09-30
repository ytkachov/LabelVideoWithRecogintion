from PyQt5.QtGui import *
from PyQt5.QtCore import *

from libs.utils import distance
import sys

DEFAULT_LINE_COLOR = QColor(200, 200, 0, 255)
DEFAULT_FILL_COLOR = QColor(200, 200, 0, 80)
LINE_WIDTH = 2
TEXT_GAP = 3
MIN_Y_LABEL = 10


class DetectedShape(object):
    # The following class variables influence the drawing
    # of _all_ shape objects.
    line_color = DEFAULT_LINE_COLOR
    fill_color = DEFAULT_FILL_COLOR
    scale = 1.0

    def __init__(self, label, fclass, score, extent):
        self.label = label
        self.score = score
        self.extent = extent
        self.selected = False
        self.visible = True
        self.fclass = fclass
        self.garbaged = False

    def paint(self, painter):
        if self.extent and self.visible:
            color = self.line_color
            pen = QPen(color)

            pen.setWidth(LINE_WIDTH)
            painter.setPen(pen)

            line_path = QPainterPath()
            line_path.moveTo(self.extent[0], self.extent[1])

            line_path.lineTo(self.extent[2], self.extent[1])
            line_path.lineTo(self.extent[2], self.extent[3])
            line_path.lineTo(self.extent[0], self.extent[3])
            line_path.lineTo(self.extent[0], self.extent[1])

            painter.drawPath(line_path)
            if self.selected:
                painter.fillPath(line_path, self.fill_color)

            # Draw text at the top-left
            min_x = self.extent[0]
            min_y = self.extent[1]

            if self.label is None:
                self.label = ""

            lbl = str.format('{2}: {0} - {1}%', self.label, str(int(self.score * 100.0)), self.fclass)

            font = QFont()
            font.setPointSize(8)
            painter.setFont(font)

            fm = QFontMetrics(font)
            w = fm.width(lbl)
            h = fm.height()

            rectf = QRectF(min_x - LINE_WIDTH, min_y - h - TEXT_GAP - LINE_WIDTH, w + TEXT_GAP, h + TEXT_GAP + LINE_WIDTH)
            rect_path = QPainterPath()
            rect_path.addRect(rectf)
            painter.fillPath(rect_path, self.line_color)

            painter.setPen(QPen(Qt.black))
            painter.drawText(rectf, Qt.AlignCenter, lbl)




