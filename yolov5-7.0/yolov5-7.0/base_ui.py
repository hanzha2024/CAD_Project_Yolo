import cv2
import sys
import torch
from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer

from ui_main_window import Ui_MainWindow


def convert2QImage(img):
    height, width, channel = img.shape
    return QImage(img, width, height, width * channel, QImage.Format_RGB888)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.model = torch.hub.load("./", "custom", path="./CAD_Detect_Models/yolov5l_based_best.pt", source="local")
        self.timer = QTimer()
        self.timer.setInterval(1)
        self.video = None
        self.bind_slots()

    def image_pred(self, file_path):
        results = self.model(file_path)
        image = results.render()[0]
        return convert2QImage(image)

    def open_image(self):
        print("点击了检测图片！")
        self.timer.stop()
        # file_path = QFileDialog.getOpenFileName(self, dir="../datasets/images/train", filter="*.jpg;*.png;*.jpeg")
        file_path = QFileDialog.getOpenFileName(self, dir="../datasets/YOLO_CAD_Dataset/val/testimage", filter="*.jpg;*.png;*.jpeg")
        if file_path[0]:
            print(file_path[0])
            print(type(file_path[0]))
            file_path = file_path[0]
            qimage = self.image_pred(file_path)
            self.input.setPixmap(QPixmap(file_path))
            self.output.setPixmap(QPixmap.fromImage(qimage))

    def bind_slots(self):
        self.detect_image.clicked.connect(self.open_image)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    
    app.exec()