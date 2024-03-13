import cv2
import numpy as np

def mark_polygons_from_image(img):
    # Read the image
    if isinstance(img, str):
        frame = cv2.imread(img)
    else:
        frame = img

    # Lists to store marked points for compartments
    compartment1_points = []
    compartment2_points = []
    current_compartment = None

    # Mouse callback function
    def mark_point(event, x, y, flags, param):
        nonlocal current_compartment
        if event == cv2.EVENT_LBUTTONDOWN:
            if current_compartment == 1:
                compartment1_points.append((x, y))
            elif current_compartment == 2:
                compartment2_points.append((x, y))
            update_visual_feedback()

    # Function to update visual feedback
    def update_visual_feedback():
        feedback_frame = frame.copy()
        for point in compartment1_points:
            cv2.circle(feedback_frame, point, 5, (0, 0, 255), -1)
        for point in compartment2_points:
            cv2.circle(feedback_frame, point, 5, (0, 255, 0), -1)
        if len(compartment1_points) > 1:
            cv2.polylines(feedback_frame, [np.array(compartment1_points)], isClosed=True, color=(0, 0, 255), thickness=2)
        if len(compartment2_points) > 1:
            cv2.polylines(feedback_frame, [np.array(compartment2_points)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imshow("Mark Points", feedback_frame)

    # Create a window
    cv2.namedWindow("Mark Points")
    # Set the mouse callback
    cv2.setMouseCallback("Mark Points", mark_point)

    while True:
        # Update and show the visual feedback
        update_visual_feedback()

        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            current_compartment = 1
        elif key == ord('2'):
            current_compartment = 2
        elif key == ord('s'):
            print("Saved Points for Compartment 1:", compartment1_points)
            # compartment1_points = []
            print("Saved Points for Compartment 2:", compartment2_points)
            # compartment2_points = []
            break
        elif key == ord('q'):
            return "Broken prematurely", "Error"

    # Destroy OpenCV windows
    cv2.destroyAllWindows()

    return compartment1_points, compartment2_points

if __name__ == "__main__":
    compartment1_points, compartment2_points = mark_polygons_from_image('/Users/nisargdoshi/Downloads/work/onm/motion_compartment/frame.png')
