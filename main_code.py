import sys
import cv2
import mediapipe as mp
import numpy as np
import struct
import socket

# Initialize server sockets

# MediaPipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Stereo camera parameters
baseline = 7  # Distance between the two cameras [cm]
f = 4  # Focal length of the camera [mm]
alpha = 60  # Camera field of view in the horizontal plane [degrees]
Mouth_THRESHOLD = 0.004

# Load stereo camera parameters
cv_file = cv2.FileStorage()
if not cv_file.open(r'C:\Users\mayan\AppData\Local\Programs\Python\Python311\opencv\depth_estimation\stereoMap3.xml', cv2.FileStorage_READ):
    print("Error: Unable to open the stereo map file.")
    sys.exit()

# Read the rectification maps
stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
cv_file.release()

def undistortRectify(frameR, frameL):
    # Undistort and rectify images
    undistortedL = cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LINEAR)
    undistortedR = cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LINEAR)
    return undistortedR, undistortedL

def find_depth(right_point, left_point, frame_right, frame_left, baseline, f, alpha):
    # Convert focal length from [mm] to [pixel]:
    height_right, width_right, _ = frame_right.shape
    height_left, width_left, _ = frame_left.shape

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)
    else:
        print('Left and right camera frames do not have the same pixel width')
        return None

    x_right = right_point[0]
    x_left = left_point[0]

    # Calculate the disparity:
    disparity = x_left - x_right  # Displacement between left and right frames [pixels]

    if disparity == 0:
        print('Disparity is zero, cannot calculate depth')
        return None

    # Calculate depth z:
    z_depth = (baseline * f_pixel) / disparity  # Depth in [cm]

    return z_depth

def calculate_lips_dist(landmarks):
    # Indices for upper and lower lips
    upper_lip_indices = [13, 14, 15, 16, 17]
    lower_lip_indices = [84, 85, 86, 87, 88]

    # Calculate the average height of upper and lower lips
    upper_lip_height = np.mean([landmarks[i].y for i in upper_lip_indices])
    lower_lip_height = np.mean([landmarks[i].y for i in lower_lip_indices])

    # Calculate the vertical distance between upper and lower lips
    lip_distance = lower_lip_height - upper_lip_height

    return lip_distance

def find_mouth_center(landmarks):
    mouth_indices = [13, 14, 15, 16, 17, 84, 85, 86, 87, 88]
    mouth_points = np.array([(landmarks[i].x, landmarks[i].y) for i in mouth_indices])
    center_x = np.mean(mouth_points[:, 0])
    center_y = np.mean(mouth_points[:, 1])
    return (center_x, center_y)

def process_camera_frame(frame, face_mesh):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            lips_dist = calculate_lips_dist(face_landmarks.landmark)
            if lips_dist > Mouth_THRESHOLD:
                cv2.putText(frame, "Opened", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Closed", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mouth_center = find_mouth_center(face_landmarks.landmark)
            mouth_center_x = int(mouth_center[0] * frame.shape[1])
            mouth_center_y = int(mouth_center[1] * frame.shape[0])
            cv2.circle(frame, (mouth_center_x, mouth_center_y), 3, (255, 0, 0), -1)
            cv2.putText(frame, f"Mouth Center: ({mouth_center_x}, {mouth_center_y})", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            return frame, (mouth_center_x, mouth_center_y)
    else:
        print("No face landmarks detected")
        return frame, None

def main():
    capL = cv2.VideoCapture(1)
    capR = cv2.VideoCapture(0)

    if not capL.isOpened() or not capR.isOpened():
        print("Error: Unable to open the cameras")
        sys.exit()

    while capL.isOpened() and capR.isOpened():
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        
        if not retL or not retR:
            break

        undistortedR, undistortedL = undistortRectify(frameR, frameL)

        frameL, left_mouth_center = process_camera_frame(frameL, face_mesh)
        frameR, right_mouth_center = process_camera_frame(frameR, face_mesh)

        if left_mouth_center and right_mouth_center:
            left_point = (left_mouth_center[0], left_mouth_center[1])
            right_point = (right_mouth_center[0], right_mouth_center[1])
                
            depth = find_depth(right_point, left_point, undistortedR, undistortedL, baseline, f, alpha)
            depth=depth/100
            if depth:
                cv2.putText(frameL, f"Depth: {depth:.2f} cm", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('FrameL', frameL)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
