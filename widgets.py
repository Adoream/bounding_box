from PyQt5 import QtCore, QtGui, QtWidgets


class ImageLabel(QtWidgets.QLabel):
    """
    负责接收鼠标事件的图像显示控件：
    - “点模式”：左键单击发出 pointAdded 信号
    - “框选模式”：左键拖拽，释放时发出 bboxFinished 信号
    - 鼠标滚轮：发出 wheelScrolled 信号，由主窗口控制 slice 切换
    """
    pointAdded = QtCore.pyqtSignal(float, float)
    bboxFinished = QtCore.pyqtSignal(float, float, float, float)
    wheelScrolled = QtCore.pyqtSignal(int)  # 正数 / 负数 表示滚动方向

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = "point"

        # 用于 bbox 橡皮筋效果
        self._origin = None
        self._rubber_band = QtWidgets.QRubberBand(
            QtWidgets.QRubberBand.Rectangle, self
        )

        # 用于“丝滑”滚轮：累积小的 delta
        self._wheel_accum = 0

        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setBackgroundRole(QtGui.QPalette.Base)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                           QtWidgets.QSizePolicy.Fixed)

    def setMode(self, mode: str):
        self.mode = mode


    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() != QtCore.Qt.LeftButton:
            return
        if self.pixmap() is None:
            return

        x = event.x()
        y = event.y()
        if x < 0 or y < 0 or x >= self.width() or y >= self.height():
            return

        if self.mode == "point":
            # 点标注：直接发送坐标
            self.pointAdded.emit(float(x), float(y))
        elif self.mode == "bbox":
            # 框选模式：记录起点，并显示橡皮筋
            self._origin = QtCore.QPoint(x, y)
            self._rubber_band.setGeometry(
                QtCore.QRect(self._origin, QtCore.QSize())
            )
            self._rubber_band.show()


    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        # 仅在 bbox 模式且已按下左键的情况下更新橡皮筋矩形
        if self.mode != "bbox":
            return
        if self._origin is None:
            return
        if self.pixmap() is None:
            return

        x = event.x()
        y = event.y()
        rect = QtCore.QRect(self._origin, QtCore.QPoint(x, y)).normalized()
        self._rubber_band.setGeometry(rect)


    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if self.mode != "bbox":
            return
        if self._origin is None:
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        if self.pixmap() is None:
            return

        # 结束橡皮筋显示
        self._rubber_band.hide()

        start = self._origin
        end = event.pos()
        self._origin = None

        x1 = float(start.x())
        y1 = float(start.y())
        x2 = float(end.x())
        y2 = float(end.y())

        self.bboxFinished.emit(x1, y1, x2, y2)


    def wheelEvent(self, event: QtGui.QWheelEvent):
        """
        将滚轮滚动转换为“步数”，并做累积：
        - 普通鼠标：通常每次 120，steps=1/−1
        - 触控板：会产生很多小于 120 的 delta，通过 self._wheel_accum 累积，
          达到一档（120）才发出一次 wheelScrolled，从而感觉更丝滑。
        """
        delta = event.angleDelta().y()
        if delta == 0:
            return

        self._wheel_accum += delta
        step_unit = 120  # 一般 120 是一“格”

        steps = int(self._wheel_accum / step_unit)
        if steps != 0:
            # 剩余的碎片保留在累积里，下一次滚动继续加
            self._wheel_accum -= steps * step_unit
            self.wheelScrolled.emit(steps)

