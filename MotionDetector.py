import cv2
import numpy as np
import random
import time
from shapely.geometry import Polygon, Point

from mark_points import mark_polygons_from_image


class MotionDetector:
    def __init__(self, no_motion_frames_threshold=5):
        self.prev_gray = None
        self.compartment1_points = None
        self.compartment2_points = None
        self.points_marked = False
        self.flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                poly_n=5, poly_sigma=1.2, flags=0)
        self.no_motion_frames_threshold = no_motion_frames_threshold
        self.motion_detected = False
        self.no_motion_frame_count = 0
        self.start_frame = None
        self.end_frame = None
        self.capture_start_frame = False
        self.scale_factor = 0.5
        self.frame_motion_threshold = 0.2
        self.compartment_motion_threshold = 0.5
        
    def process_frame(self, frame):
        small_frame = cv2.resize(frame, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_AREA)
        current_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = current_gray
            return

        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, current_gray, None, **self.flow_params)
        avg_motion_compartment = self.calculate_avg_motion(flow, current_gray, self.scale_factor)

        self.update_motion_status(avg_motion_compartment, frame)
        cv2.polylines(frame, [self.compartment1_points], True, (255, 0, 0), 2)
        cv2.polylines(frame, [self.compartment2_points], True, (0, 255, 0), 2)
        self.prev_gray = current_gray
        cv2.imshow('Motion Detection', frame)

    def calculate_avg_motion(self, flow):
        motion_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        avg_motion = np.mean(motion_mag)  # Calculate the mean motion magnitude for the entire frame
        return(avg_motion)

    def update_motion_status(self, avg_motion_compartment, frame):
        # print(avg_motion_compartment)
        if avg_motion_compartment > self.frame_motion_threshold:
            if not self.motion_detected:
                self.motion_detected = True
                self.no_motion_frame_count = 0
            else:
                self.no_motion_frame_count = 0
        else:
            if self.motion_detected:
                self.no_motion_frame_count += 1                

        # print(self.no_motion_frame_count)

        if self.motion_detected and self.no_motion_frame_count >= self.no_motion_frames_threshold:
            self.end_frame = frame.copy()
            self.analyze_motion(frame)
            self.motion_detected = False
            self.capture_start_frame = False

    def analyze_motion(self, frame):
        start_gray = cv2.cvtColor(self.start_frame, cv2.COLOR_BGR2GRAY)
        end_gray = cv2.cvtColor(self.end_frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("previous_frame.jpg", start_gray)
        cv2.imwrite("still_frame.jpg", end_gray)

        flow = cv2.calcOpticalFlowFarneback(start_gray, end_gray, None, **self.flow_params)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        motion_in_compartments = []
        avg_flow_magnitudes = []  
        for compartment_points in [self.compartment1_points, self.compartment2_points]:
            mask = np.zeros_like(start_gray, dtype=np.uint8)
            cv2.fillPoly(mask, [compartment_points], 255)

            compartment_flow_mag = cv2.bitwise_and(magnitude, magnitude, mask=mask)

            avg_flow_magnitude = np.sum(compartment_flow_mag) / cv2.countNonZero(mask)
            avg_flow_magnitudes.append(avg_flow_magnitude)

            motion_in_compartments.append(avg_flow_magnitude > self.compartment_motion_threshold)  # Adjust the threshold as needed
        print(motion_in_compartments)
        print(avg_flow_magnitudes)

        if motion_in_compartments[0] and motion_in_compartments[1]:
            if avg_flow_magnitudes[0] > avg_flow_magnitudes[1]:
                print("More significant motion in compartment 1")
                self.start_frame = frame.copy()
                print("Saved")
                cv2.fillPoly(frame, pts=[self.compartment1_points], color=(255, 0, 255))
            else:  # Assumes motion in compartment 2 is equal or more significant
                print("More significant motion in compartment 2")
                self.start_frame = frame.copy()
                print("Saved")
                cv2.fillPoly(frame, pts=[self.compartment2_points], color=(255, 0, 255))
        else:
            # Handle individual compartment motion detection with significant average flow magnitude
            if avg_flow_magnitudes[0] and avg_flow_magnitudes[0] > self.compartment_motion_threshold:
                print("Motion in compartment 1")
                self.start_frame = frame.copy()
                print("Saved")
                cv2.fillPoly(frame, pts=[self.compartment1_points], color=(0, 0, 255))
            elif motion_in_compartments[1] and avg_flow_magnitudes[1] > self.compartment_motion_threshold:
                print("Motion in compartment 2") 
                self.start_frame = frame.copy()
                print("Saved")
                cv2.fillPoly(frame, pts=[self.compartment2_points], color=(0, 0, 255))

    def run(self):
        cap = cv2.VideoCapture(3)
        c = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if not self.points_marked:
                cv2.imwrite("frame.png", frame)
                self.compartment1_points, self.compartment2_points = mark_polygons_from_image(frame)
                # self.compartment1_points = [(221, 24), (73, 39), (88, 234), (163, 322), (289, 230), (335, 206), (376, 14)]
                # self.compartment2_points =  [(384, 15), (352, 205), (235, 274), (175, 305), (169, 333), (307, 477), (354, 479), (441, 479), (618, 220), (637, 169), (637, 66)]
                
                self.compartment1_polygon = Polygon(self.compartment1_points)
                self.compartment2_polygon = Polygon(self.compartment2_points)
                self.compartment1_points = np.array([self.compartment1_points], dtype=np.int32 )
                self.compartment2_points = np.array([self.compartment2_points], dtype=np.int32 )
                self.points_marked = True

            self.process_frame(frame)
            c+=1
            if c==1:
                self.start_frame = frame.copy()
                print("Saved")
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

motion_detector = MotionDetector()
motion_detector.run()
