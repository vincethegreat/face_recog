import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QApplication, QFileDialog, QLabel, QMainWindow
from simple_facerec import SimpleFacerec
import sys

# e encode ang mga nawong sa folders
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create a label to display the video feed
        self.label = QLabel(self)
        self.label.setGeometry(QtCore.QRect(10, 10, 480, 360))

        # Load Camera
        self.cap = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)
        #Submit Image button
        self.open_folder_button = QPushButton('Submit Image', self)
        self.open_folder_button.setGeometry(QtCore.QRect(10, 380, 100, 30))
        self.open_folder_button.clicked.connect(self.open_folder)
        # Create a label to display the logo
        self.logo_label = QLabel(self)
        self.logo_label.setGeometry(QtCore.QRect(self.width() - 110, 10, 100, 100))
        self.logo_label.setAlignment(QtCore.Qt.AlignRight)
        # Load the logo image
        logo_image = QPixmap("logo/logo.png")
        self.logo_label.setPixmap(logo_image)

        # Make the logo label transparent
        self.logo_label.setAttribute(QtCore.Qt.WA_TranslucentBackground)

    def open_folder(self):
        # Function to be called when open folder button is clicked
        folder_path = QUrl.fromLocalFile('C:/Users/user/repos/face-recognition-for-wanted-criminals/face_recog/images')
        QDesktopServices.openUrl(folder_path)

    def update_frame(self):
        ret, frame = self.cap.read()
        self.setWindowTitle("Image Processing for Wanted Criminals")
        icon = QIcon("icon/icon.png")  # Load the icon image
        self.setWindowIcon(icon)

        # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        # Convert the frame to a QImage and set it to the label
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qImg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(qImg))

    def closeEvent(self, event):
        self.cap.release()

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
