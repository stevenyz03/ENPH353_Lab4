#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi
import numpy as np
import cv2
import sys

class My_App(QtWidgets.QMainWindow):
    """
    Main application class for the SIFT-based image processing app.
    """

    def __init__(self):
        """
        Initialize the application, set up the UI, camera, and SIFT processor.
        """
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        # Camera setup
        self._cam_id = 0
        self._cam_fps = 2
        self._is_cam_enabled = False

        # SIFT processor
        self.sift = cv2.SIFT_create()
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=0), dict())

        # Connect UI buttons to their functions
        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        # Initialize camera device
        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)  # Set width
        self._camera_device.set(4, 240)  # Set height

        # Timer for fetching camera frames
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)

        # Flag to check if image is loaded
        self.img_loaded = False

    def SLOT_browse_button(self):
        """
        Slot function to handle the browse button click.
        Opens a file dialog to select an image and loads it as a template.
        """
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]
            self.img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
            self.kp_target, self.desc_target = self.sift.detectAndCompute(self.img, None)
            self.img_loaded = True

            pixmap = QtGui.QPixmap(self.template_path)
            self.template_label.setPixmap(pixmap)
            print("Loaded template image file: " + self.template_path)

    def convert_cv_to_pixmap(self, cv_img):
        """
        Converts an OpenCV image to a QPixmap for display in the GUI.
        :param cv_img: The OpenCV image to convert.
        :return: QPixmap object.
        """
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        """
        Slot function to handle camera frame updates.
        Captures a frame from the camera, processes it, and updates the GUI.
        """
        ret, frame = self._camera_device.read()
        if not ret:
            print("Failed to capture frame")
            return

        processed_frame = self.sifting(frame) if self.img_loaded else frame
        pixmap = self.convert_cv_to_pixmap(processed_frame)
        self.live_image_label.setPixmap(pixmap)

    def SLOT_toggle_camera(self):
        """
        Slot function to handle camera toggle button.
        Starts or stops the camera capture based on the current state.
        """
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
            self.live_image_label.setText("Camera is Off")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")

    def sifting(self, frame):
        """
        Processes the given frame using SIFT to find and highlight the template.
        :param frame: The frame to process.
        :return: Processed frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kp, descriptors = self.sift.detectAndCompute(gray, None)

        matches = self.flann.knnMatch(self.desc_target, descriptors, k=2)
        matching_kp = [i for i, j in matches if i.distance < 0.6 * j.distance]

        if len(matching_kp) >= 4:
            img_points = np.float32([self.kp_target[i.queryIdx].pt for i in matching_kp]).reshape(-1, 1, 2)
            frame_points = np.float32([kp[i.trainIdx].pt for i in matching_kp]).reshape(-1, 1, 2)

            matrix, _ = cv2.findHomography(img_points, frame_points, cv2.RANSAC, 5.0)
            h, w = self.img.shape
            corner_pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            transformed_pts = cv2.perspectiveTransform(corner_pts, matrix)
            processed_frame = cv2.polylines(frame, [np.int32(transformed_pts)], True, (0, 0, 255), 3)
        else:
            processed_frame = frame

        return processed_frame

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())
