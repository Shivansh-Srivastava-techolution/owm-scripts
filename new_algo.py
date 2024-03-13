import cv2
import numpy as np

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

    def process_frame(self, frame):
        scale_factor = 0.5
        small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        current_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = current_gray
            return

        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, current_gray, None, **self.flow_params)
        cv2.polylines(frame, [self.compartment1_points], True, (255, 0, 0), 2)
        cv2.polylines(frame, [self.compartment2_points], True, (0, 255, 0), 2)
        avg_motion_compartment = self.calculate_avg_motion(flow, current_gray, scale_factor)

        self.update_motion_status(avg_motion_compartment, frame)
        self.prev_gray = current_gray
        cv2.imshow('Motion Detection', frame)

    def calculate_avg_motion(self, flow, current_gray, scale_factor):
        avg_motion_compartment = []
        for points in [self.compartment1_points, self.compartment2_points]:
            mask = np.zeros_like(current_gray)
            cv2.fillPoly(mask, [np.int32(points * scale_factor)], 1)
            motion_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            avg_motion = np.sum(motion_mag * mask) / np.sum(mask)
            avg_motion_compartment.append(avg_motion)
        return avg_motion_compartment

    def update_motion_status(self, avg_motion_compartment, frame):
        print(avg_motion_compartment)
        if any(motion > 0.5 for motion in avg_motion_compartment):
            if not self.motion_detected:
                self.start_frame = frame.copy()
                self.motion_detected = True
                self.no_motion_frame_count = 0
            else:
                self.no_motion_frame_count = 0
        else:
            if self.motion_detected:
                self.no_motion_frame_count += 1
                
        if self.motion_detected and self.no_motion_frame_count >= self.no_motion_frames_threshold:
            self.end_frame = frame.copy()
            self.analyze_motion(frame)
            self.motion_detected = False
            self.capture_start_frame = False

    def analyze_motion(self, frame):
        start_gray = cv2.cvtColor(self.start_frame, cv2.COLOR_BGR2GRAY)
        end_gray = cv2.cvtColor(self.end_frame, cv2.COLOR_BGR2GRAY)

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

            motion_in_compartments.append(avg_flow_magnitude > 0.5)  # Adjust the threshold as needed

        if motion_in_compartments[0] and motion_in_compartments[1]:
            if avg_flow_magnitudes[0] > avg_flow_magnitudes[1]:
                print("More significant motion in compartment 1")
                cv2.fillPoly(frame, pts=[self.compartment1_points], color=(255, 255, 0))
            elif avg_flow_magnitudes[1] > avg_flow_magnitudes[0]:
                print("More significant motion in compartment 2")
                cv2.fillPoly(frame, pts=[self.compartment2_points], color=(255, 255, 0))
            else:
                print("No significant motion in both compartments")
        elif motion_in_compartments[0]:
            print("Motion in compartment 1")
            cv2.fillPoly(frame, pts=[self.compartment1_points], color=(0, 0, 255))
        elif motion_in_compartments[1]:
            print("Motion in compartment 2")
            cv2.fillPoly(frame, pts=[self.compartment2_points], color=(0, 0, 255))
        
    def run(self):
        cap = cv2.VideoCapture(r"input\0OmQ7ta4.mp4")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if not self.points_marked:
                cv2.imwrite("frame.png", frame)
                self.compartment1_points, self.compartment2_points = mark_polygons_from_image(frame)
                # self.compartment1_points = [(355, 9), (354, 87), (352, 177), (337, 216), (288, 247), (236, 282), (193, 317), (167, 337), (127, 304), (99, 253), (74, 222), (67, 108), (57, 50), (57, 37), (157, 11), (212, 2), (361, 2)]
                # self.compartment2_points = [(370, 4), (362, 21), (360, 87), (357, 177), (341, 220), (197, 321), (171, 345), (225, 408), (269, 451), (296, 479), (393, 477), (441, 477), (517, 395), (601, 255), (638, 167), (638, 41), (636, 5)]
                self.compartment1_points = np.array([self.compartment1_points], dtype=np.int32)
                self.compartment2_points = np.array([self.compartment2_points], dtype=np.int32)
                self.points_marked = True

            self.process_frame(frame)

            if cv2.waitKey(2) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    motion_detector = MotionDetector()
    motion_detector.run()