import cv2
import numpy as np
import random
import time
from shapely.geometry import Polygon, Point

from mark_points import mark_polygons_from_image


class MotionDetector:
    def __init__(self, alpha=0.10):
        self.alpha = alpha
        self.prev_frame = None
        self.all_detected_points = []
        self.display_counter = 0
        self.display_limit = 0
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
        self.out = None
        self.recording = False
        self.initial_motion_frame = None
        self.previous_still_frame = None
        self.current_still_frame = None
        self.motion_done = False 
        self.motion_flag = False
        self.points_marked = False
        self.motion_in_which_compartment = ""
        self.percentage_mask_1 = 0
        self.percentage_mask_2 = 0
        self.flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                poly_n=5, poly_sigma=1.2, flags=0)

    def find_convex_hull_center(self, contour):
        hull = cv2.convexHull(contour)
        M = cv2.moments(hull)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
        else:
            return None

    def check_contour_compartment(self, contour, compartment1_polygon, compartment2_polygon):
        if compartment1_polygon.contains(Point(contour[0])):
            return 1
        elif compartment2_polygon.contains(Point(contour[0])):
            return 2
        else:
            return None

    def process_frame(self, frame):
        frame_copy = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray.astype(float)
            return

        cv2.accumulateWeighted(gray, self.prev_frame, self.alpha)
        avg_frame = cv2.convertScaleAbs(self.prev_frame)
        frame_diff = cv2.absdiff(gray, avg_frame)
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.prev_frame = gray.astype('float').copy()

        self.all_detected_points = []

        if len(contours) != 0:
            for contour in contours:
                if cv2.contourArea(contour) > 200:
                    self.all_detected_points.extend(contour.reshape(-1, 2))
                compartment_number = self.check_contour_compartment(contour, self.compartment1_polygon, self.compartment2_polygon)
                if compartment_number is not None:
                    self.motion_in_which_compartment = str(compartment_number)
                    self.previous_still_frame = gray.astype('float').copy()
                    self.motion_done = True
                    self.motion_flag = True     
                else:
                    self.motion_in_which_compartment = "None"
        else:
            self.motion_done = False       
        
        if self.all_detected_points:
            # with open('all_detected_points.txt', 'w') as f:
            #     for item in self.all_detected_points:
            #         f.write("%s\n" % item)

            self.all_detected_points = np.asarray(self.all_detected_points)
            hull = cv2.convexHull(self.all_detected_points)
            coords = self.find_convex_hull_center(hull)
            if coords is None:
                cx, cy = 0, 0
            else:
                cx, cy = coords
            cv2.drawContours(frame, [hull], 0, (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), radius=15, color=(0, 0, 255), thickness=-1)

        else:
            if self.previous_still_frame is None:
                self.previous_still_frame = gray.astype(float)
            else:
                if not self.motion_done and self.motion_flag:
                    self.current_still_frame = gray.astype(float)
                    cv2.imwrite("still_frame.jpg", self.current_still_frame)
                    cv2.imwrite("previous_frame.jpg", self.previous_still_frame)
                    flow = cv2.calcOpticalFlowFarneback(self.previous_still_frame, self.current_still_frame, None, **self.flow_params)
                    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                    motion_in_compartments = []
                    avg_flow_magnitudes = []  
                    for compartment_points in [self.compartment1_points, self.compartment2_points]:
                        mask = np.zeros_like(self.previous_still_frame, dtype=np.uint8)
                        cv2.fillPoly(mask, [compartment_points], 255)

                        compartment_flow_mag = cv2.bitwise_and(magnitude, magnitude, mask=mask)

                        avg_flow_magnitude = np.sum(compartment_flow_mag) / cv2.countNonZero(mask)
                        avg_flow_magnitudes.append(avg_flow_magnitude)

                        motion_in_compartments.append(avg_flow_magnitude > 0.5)  # Adjust the threshold as needed
                    print(motion_in_compartments)
                    print(avg_flow_magnitudes)

                    if motion_in_compartments[0] and motion_in_compartments[1]:
                        if avg_flow_magnitudes[0] > avg_flow_magnitudes[1]:
                            print("More significant motion in compartment 1")
                            cv2.fillPoly(frame, pts=[self.compartment1_points], color=(255, 0, 255))
                        else:  # Assumes motion in compartment 2 is equal or more significant
                            print("More significant motion in compartment 2")
                            cv2.fillPoly(frame, pts=[self.compartment2_points], color=(255, 0, 255))
                    else:
                        # Handle individual compartment motion detection with significant average flow magnitude
                        if avg_flow_magnitudes[0] and avg_flow_magnitudes[0] > 0.1:
                            print("Motion in compartment 1")
                            cv2.fillPoly(frame, pts=[self.compartment1_points], color=(0, 0, 255))
                        elif motion_in_compartments[1] and avg_flow_magnitudes[1] > 0.1:
                            print("Motion in compartment 2") 
                            cv2.fillPoly(frame, pts=[self.compartment2_points], color=(0, 0, 255))
                    self.motion_flag = False
                        
        cv2.polylines(frame, [self.compartment1_points], True, (255, 0, 0), 2)
        cv2.polylines(frame, [self.compartment2_points], True, (0, 255, 0), 2)
        
        cv2.imshow('Motion Detection', frame)

    def run(self):
        cap = cv2.VideoCapture(r"input\0OmQ7ta4.mp4")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if not self.points_marked:
                cv2.imwrite("frame.png", frame)
                # self.compartment1_points, self.compartment2_points = mark_polygons_from_image(frame)
                self.compartment1_points = [(221, 24), (73, 39), (88, 234), (163, 322), (289, 230), (335, 206), (376, 14)]
                self.compartment2_points =  [(384, 15), (352, 205), (235, 274), (175, 305), (169, 333), (307, 477), (354, 479), (441, 479), (618, 220), (637, 169), (637, 66)]
                
                self.compartment1_polygon = Polygon(self.compartment1_points)
                self.compartment2_polygon = Polygon(self.compartment2_points)
                self.compartment1_points = np.array([self.compartment1_points], dtype=np.int32 )
                self.compartment2_points = np.array([self.compartment2_points], dtype=np.int32 )
                self.points_marked = True

            self.process_frame(frame)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

motion_detector = MotionDetector()
motion_detector.run()
