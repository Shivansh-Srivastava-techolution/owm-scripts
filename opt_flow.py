import cv2
import numpy as np
from shapely.geometry import Polygon

class MotionDetector:
    def __init__(self, alpha=0.10):
        self.alpha = alpha
        self.prev_gray = None
        self.compartment1_points = None
        self.compartment2_points = None
        self.points_marked = False
        self.flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                poly_n=5, poly_sigma=1.2, flags=0)
    
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

        avg_motion_compartment = []
        for points in [self.compartment1_points, self.compartment2_points]:
            mask = np.zeros_like(current_gray)
            cv2.fillPoly(mask, [np.int32(points * scale_factor)], 1)
            motion_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            avg_motion = np.sum(motion_mag * mask) / np.sum(mask)
            avg_motion_compartment.append(avg_motion)

        if avg_motion_compartment[0] > avg_motion_compartment[1] and avg_motion_compartment[0] > 0.5:
            cv2.fillPoly(frame, pts=[self.compartment1_points], color=(0, 0, 255))
        elif avg_motion_compartment[1] > 0.5:
            cv2.fillPoly(frame, pts=[self.compartment2_points], color=(0, 0, 255))

        self.prev_gray = current_gray
        cv2.imshow('Motion Detection', frame)

    def run(self):
        cap = cv2.VideoCapture(r"input\dTxl3C6H.mp4")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if not self.points_marked:
                cv2.imwrite("frame.png", frame)
                # Assign your compartment points here, the hardcoded values are placeholders
                # Example compartment points:
                self.compartment1_points = [(355, 9), (354, 87), (352, 177), (337, 216), (288, 247), (236, 282), (193, 317), (167, 337), (127, 304), (99, 253), (74, 222), (67, 108), (57, 50), (57, 37), (157, 11), (212, 2), (361, 2)]
                self.compartment2_points = [(370, 4), (362, 21), (360, 87), (357, 177), (341, 220), (197, 321), (171, 345), (225, 408), (269, 451), (296, 479), (393, 477), (441, 477), (517, 395), (601, 255), (638, 167), (638, 41), (636, 5)]
                self.compartment1_points = np.array([self.compartment1_points], dtype=np.int32)
                self.compartment2_points = np.array([self.compartment2_points], dtype=np.int32)
                self.points_marked = True

            self.process_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    motion_detector = MotionDetector()
    motion_detector.run()