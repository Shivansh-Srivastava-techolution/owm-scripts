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
        self.previous_still_frame = None
        self.current_still_frame = None
        self.motion_done = False 
        self.points_marked = False
        self.motion_in_which_compartment = ""
        self.percentage_mask_1 = 0
        self.percentage_mask_2 = 0

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
                    self.motion_done = True
                else:
                    self.motion_in_which_compartment = "None"
        
        if self.all_detected_points:
            with open('all_detected_points.txt', 'w') as f:
                for item in self.all_detected_points:
                    f.write("%s\n" % item)

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
                if self.motion_done:
                    self.current_still_frame = gray.astype(float)
                    still_diff = cv2.absdiff(self.previous_still_frame, self.current_still_frame)
                    psf_write = self.previous_still_frame.copy()
                    cv2.imwrite("previous_still_frame.png", psf_write)
                    csf_write = self.current_still_frame.copy()
                    cv2.imwrite("current_still_frame.png", csf_write)
                    sd_write = still_diff.copy()
                    cv2.imwrite("still_diff.png", sd_write)
                    self.previous_still_frame = self.current_still_frame
                    still_diff[still_diff>=1] = 1
                    still_diff[still_diff<1] = 0
                    compartment1_points_mask = np.zeros_like(still_diff, dtype=np.uint8)
                    compartment2_points_mask = np.zeros_like(still_diff, dtype=np.uint8)
                    cv2.fillPoly(compartment1_points_mask, [self.compartment1_points], 1)
                    cv2.fillPoly(compartment2_points_mask, [self.compartment2_points], 1)
                    cv2.imwrite("compartment1.png", compartment1_points_mask*255)
                    cv2.imwrite("compartment2.png", compartment2_points_mask*255)
                    still_diff = still_diff.astype(np.uint8)
                    final_1 = cv2.bitwise_and(still_diff, compartment1_points_mask)
                    final_2 = cv2.bitwise_and(still_diff, compartment2_points_mask)
                    cv2.imwrite("final1.png", final_1*255)
                    cv2.imwrite("final2.png", final_2*255)
                    compartment1_points_sum = int(np.sum(final_1))
                    compartment2_points_sum = int(np.sum(final_2))
                    self.percentage_mask_1 = compartment1_points_sum / len(np.where(compartment1_points_mask==1)[0])
                    self.percentage_mask_2 = compartment2_points_sum / len(np.where(compartment2_points_mask==1)[0])
                    if compartment1_points_sum < compartment2_points_sum:
                        percentage_change = abs(compartment1_points_sum - compartment2_points_sum) / (compartment1_points_sum+1) * 100
                    else:
                        percentage_change = abs(compartment1_points_sum - compartment2_points_sum) / (compartment2_points_sum+1) * 100
                    print("-"*50)
                    print(" Compartment-1 Sum:", compartment1_points_sum)
                    print(" Compartment-2 Sum:", compartment2_points_sum)
                    print(" => Percentage change: ", percentage_change)
                    print("-"*50)
                    if compartment1_points_sum > compartment2_points_sum:
                        cv2.fillPoly(frame, pts=[self.compartment1_points], color=(0, 0, 255))
                        print("=== C1 ===")
                    if compartment1_points_sum < compartment2_points_sum:
                        cv2.fillPoly(frame, pts=[self.compartment2_points], color=(0, 0, 255))
                        print("=== C2 ===")
                    elif compartment1_points_sum == compartment2_points_sum:
                        if compartment1_points_sum==0 and compartment2_points_sum==0:
                            print("No change in compartments")
                        elif compartment1_points_sum!=0:
                            print("Type: ", type(compartment2_points_sum))
                            print("Sums: ", compartment1_points_sum, compartment2_points_sum)
                            print("Equal change in both compartments which is not zero")
        cv2.polylines(frame, [self.compartment1_points], True, (255, 0, 0), 2)
        cv2.polylines(frame, [self.compartment2_points], True, (0, 255, 0), 2)
        if self.percentage_mask_1 > 50:
            cv2.putText(frame, "Item added/removed 1", (50,50), cv2.FONT_HERSHEY_SIMPLEX , 2, (0,255,0), 5 )
        elif self.percentage_mask_2 > 50:
            cv2.putText(frame, "Item added/removed 2", (50,50), cv2.FONT_HERSHEY_SIMPLEX , 2, (0,255,0), 5 )
        elif self.percentage_mask_1>50 and self.percentage_mask_2>50:
            cv2.putText(frame, "Both more than 50", (50,50), cv2.FONT_HERSHEY_SIMPLEX , 2, (0,255,0), 5 )
        cv2.imshow('Motion Detection', frame)

    def start_recording(self):
        self.out = cv2.VideoWriter(f'output/output_{random.randint(1, 9999)}.mp4', self.fourcc, 10.0, (640, 480))
        self.recording = True

    def stop_recording(self):
        self.out.release()
        self.recording = False

    def run(self):
        cap = cv2.VideoCapture("input/N28IwSDc.mp4")
        print("Resolution: ",cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("FPS: ",cap.get(cv2.CAP_PROP_FPS))

        while True:
            ret, frame = cap.read()
            if ret == False:
                break

            if not self.points_marked:
                cv2.imwrite("frame.png", frame)
                # self.compartment1_points, self.compartment2_points = mark_polygons_from_image(frame)
                self.compartment1_points = [(355, 9), (354, 87), (352, 177), (337, 216), (288, 247), (236, 282), (193, 317), (167, 337), (127, 304), (99, 253), (74, 222), (67, 108), (57, 50), (57, 37), (157, 11), (212, 2), (361, 2)]
                self.compartment2_points = [(370, 4), (362, 21), (360, 87), (357, 177), (341, 220), (197, 321), (171, 345), (225, 408), (269, 451), (296, 479), (393, 477), (441, 477), (517, 395), (601, 255), (638, 167), (638, 41), (636, 5)]

                if isinstance(self.compartment1_points, str):
                    break

                self.compartment1_polygon = Polygon(self.compartment1_points)
                self.compartment2_polygon = Polygon(self.compartment2_points)
                self.compartment1_points = np.array([self.compartment1_points], dtype=np.int32 )
                self.compartment2_points = np.array([self.compartment2_points], dtype=np.int32 )

                self.points_marked = True

            self.process_frame(frame)

            key = cv2.waitKey(2) & 0xFF

            if key == ord('r') and not self.recording:
                print("Recording Started")
                self.start_recording()

            elif key == ord('s') and self.recording:
                print("Recording Stopped")
                self.stop_recording()

            if self.recording:
                self.out.write(frame)

            elif key == ord('q'):
                break

        cap.release()
        if self.recording:
            self.out.release()
        cv2.destroyAllWindows()

motion_detector = MotionDetector()
motion_detector.run()
