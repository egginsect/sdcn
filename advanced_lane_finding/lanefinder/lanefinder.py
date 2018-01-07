import numpy as np
import cv2
import glob
class LaneFinder(object):
    def __init__(self, calibrate_img_files='camera_cal/calibration*.jpg'):
        self.compute_camera_calibration(calibrate_img_files)
        self.src = np.float32([[550, 500], [780, 500], [1100, 710], [240, 710]])
        self.dst = np.float32([[240, 100], [1080, 100], [1100, 710], [220, 710]])

    def compute_camera_calibration(self, calibrate_img_files):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(calibrate_img_files)
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)

    def preproc(self, img, thcolor, thgrad):
        undistorted = self.cal_undistort(img)
        color_thresh_mask = self.color_thresh(undistorted, thcolor)
        grad_thresh_mask = self.grad_thresh(color_thresh_mask, thgrad)
        return np.dstack([grad_thresh_mask, grad_thresh_mask, grad_thresh_mask])

    def cal_undistort(self, img):
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undist

    def corners_unwarp(self, undistorted):

        undistorted_gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        img_size = undistorted_gray.shape[::-1]

        M = cv2.getPerspectiveTransform(self.src, self.dst)
        warped = cv2.warpPerspective(undistorted, M, img_size)
        cv2.polylines(undistorted,[self.src.astype(np.int32)],True,(0,255,255))
        cv2.polylines(warped,[self.dst.astype(np.int32)],True,(0,255,255))
        return warped
    
    def grad_thresh(self, s_channel, threshod=(30, 100)):
        sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        return np.logical_and(scaled_sobel >= threshod[0], scaled_sobel <= threshod[1]).astype(np.uint8)*255

    def color_thresh(self, img, threshod=(50, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        s_channel = hls[:,:,2]
        return np.logical_and(s_channel >= threshod[0], s_channel <= threshod[1]).astype(np.uint8)*255