import cv2
import numpy as np
import time
from collections import deque
from scipy.spatial import distance as dist

# --- CONFIGURATION ---
FRAME_WIDTH = 600
FRAME_HEIGHT = 400
COUNTING_LINE_Y = int(FRAME_HEIGHT * 0.5) # Virtual horizontal counting line at 50% height
MIN_DISTANCE = 50 # Minimum distance for a new detection to be considered the same object

# --- CLASSES ---

class CentroidTracker:
    """
    A simple centroid tracking class. Assigns a unique ID to each object
    and maintains its location over time.
    """
    def __init__(self, max_disappeared=50):
        # Stores the next available unique object ID
        self.next_object_id = 0
        # Maps unique object ID to a TrackableObject instance
        self.objects = {}
        # Maps unique object ID to the number of consecutive frames it was not detected
        self.disappeared = {}
        # The number of consecutive frames an object must be missed before deregistering it
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        """Register a new object."""
        new_obj = TrackableObject(self.next_object_id, centroid)
        self.objects[self.next_object_id] = new_obj
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Deregister an old object."""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        """
        Takes a list of new bounding box rectangles (x, y, w, h) and updates the tracker.
        The core challenge of matching old and new objects happens here.
        """
        # If no detections are provided, increment disappeared count for all existing objects
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Compute centroids for the new detections
        input_centroids = []
        for (x, y, w, h) in rects:
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0)
            input_centroids.append((cX, cY))

        # Handle the first frame or if the object list is currently empty
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [obj.centroid for obj in self.objects.values()]

            # Compute the distance between each existing object centroid and each new input centroid
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # Find the minimum distance for each existing object
            rows = D.min(axis=1).argsort()
            # Find the index of the new input centroid corresponding to that minimum distance
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                # If we have already examined the row or column index, ignore it
                if row in used_rows or col in used_cols:
                    continue

                # If the distance is less than the minimum threshold, match them
                if D[row, col] < MIN_DISTANCE:
                    object_id = object_ids[row]
                    # Update the tracked object's properties
                    self.objects[object_id].update_centroid(input_centroids[col])
                    self.disappeared[object_id] = 0

                    used_rows.add(row)
                    used_cols.add(col)

            # Handle objects that were matched (rows) and not matched (disappeared)
            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            # Handle disappeared (unmatched existing) objects
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Handle newly detected (unmatched input) objects
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

class TrackableObject:
    """
    Stores an object's state: ID, history of centroids, and count status.
    """
    def __init__(self, object_id, centroid):
        self.object_id = object_id
        self.centroid = centroid
        # Store a history of past centroids for line-crossing checks
        self.centroids = deque([centroid], maxlen=5)
        # Has the object been counted?
        self.counted = False
        # Last known vertical position for directional tracking
        self.last_y = centroid[1]

    def update_centroid(self, new_centroid):
        """Update the centroid and history."""
        self.last_y = self.centroid[1]
        self.centroid = new_centroid
        self.centroids.append(new_centroid)


# --- CORE COUNTING LOGIC ---

def count_object(obj, counting_line_y, direction):
    """
    Checks if a trackable object has crossed the counting line in the specified direction.
    - direction: 'up' or 'down'
    """
    # Requires at least two points in history to determine movement direction
    if len(obj.centroids) < 2 or obj.counted:
        return False

    # Get the current and previous Y positions
    current_y = obj.centroid[1]
    prev_y = obj.centroids[-2][1]

    if direction == 'down':
        # Crossing from above to below the line (e.g., crossing down)
        if prev_y < counting_line_y and current_y >= counting_line_y:
            obj.counted = True
            return True
    elif direction == 'up':
        # Crossing from below to above the line (e.g., crossing up)
        if prev_y > counting_line_y and current_y <= counting_line_y:
            obj.counted = True
            return True

    return False

# --- SIMULATION FUNCTIONS ---
def simulate_detections(frame_count):
    """
    Simulates bounding box detections (x, y, w, h) for a few objects
    moving across the frame to trigger the counting logic.
    """
    rects = []
    # Simulate three objects moving downward
    # Object 1 (Moves across the line)
    x1, y1 = 50, frame_count * 2
    w1, h1 = 40, 40
    if y1 < FRAME_HEIGHT:
        rects.append((x1, y1, w1, h1))

    # Object 2 (Moves and stops before the line)
    x2, y2 = 250, frame_count * 1
    w2, h2 = 30, 30
    if y2 < COUNTING_LINE_Y - 50:
        rects.append((x2, y2, w2, h2))

    # Object 3 (Starts just above the line and moves slowly)
    x3, y3 = 400, COUNTING_LINE_Y - 30 + frame_count * 0.5
    w3, h3 = 50, 50
    if y3 < FRAME_HEIGHT:
        rects.append((x3, y3, w3, h3))

    return rects

# --- MAIN EXECUTION ---

def run_object_counter():
    """
    Main loop to run the simulated object counter.
    """
    # Initialize the tracker and counters
    tracker = CentroidTracker()
    count_down = 0
    count_up = 0
    frame_count = 0

    print("Starting Object Counting Simulation...")
    print(f"Counting line is at Y={COUNTING_LINE_Y} (50% of frame height)")

    while True:
        frame_count += 1
        # --- 1. SIMULATE FRAME CAPTURE ---
        # Create a blank black image (simulated video frame)
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype="uint8")

        # --- 2. DRAW VIRTUAL COUNTING LINE ---
        cv2.line(frame, (0, COUNTING_LINE_Y), (FRAME_WIDTH, COUNTING_LINE_Y), (0, 255, 255), 2)
        cv2.putText(frame, "COUNTING LINE", (10, COUNTING_LINE_Y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # --- 3. SIMULATE DETECTION (Replace with actual YOLO/ML Inference for non-simulated detections) ---
        rects = simulate_detections(frame_count)

        # --- 4. UPDATE TRACKER AND PERFORM COUNTING ---
        objects = tracker.update(rects)

        for (object_id, obj) in objects.items():
            # Check for crossing in the 'down' direction
            if not obj.counted and count_object(obj, COUNTING_LINE_Y, 'down'):
                count_down += 1
                # Mark the object as counted
                obj.counted = True

            # Check for crossing in the 'up' direction (optional, but good for completeness)
            if not obj.counted and count_object(obj, COUNTING_LINE_Y, 'up'):
                count_up += 1
                obj.counted = True

            # --- 5. VISUALIZATION (Draw Bounding Box and ID) ---
            # Simulate drawing the bounding box and centroid (simplified for demo)
            cX, cY = obj.centroid
            # Draw centroid
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            # Draw ID
            text = f"ID {object_id}"
            color = (0, 255, 0) if obj.counted else (255, 255, 255)
            cv2.putText(frame, text, (cX - 10, cY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- 6. DISPLAY COUNTS ---
        info_down = f"Down Count: {count_down}"
        info_up = f"Up Count: {count_up}"
        cv2.putText(frame, info_down, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, info_up, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Display the output frame
        cv2.imshow("Object Counter Demo", frame)

        # Break the loop on 'q' key press or if simulation ends
        if cv2.waitKey(100) & 0xFF == ord('q') or frame_count > 250:
            break

        # Simulate real-time delay
        time.sleep(0.02) # About 50 FPS simulation speed

    cv2.destroyAllWindows()
    print("Simulation finished.")
    print(f"Final Count Down: {count_down}")
    print(f"Final Count Up: {count_up}")

if __name__ == '__main__':
    # Add this check to ensure the script can run if not in an environment with a GUI
    try:
        run_object_counter()
    except Exception as e:
        print(f"An error occurred (often due to missing OpenCV GUI environment): {e}")
        print("Run this script in a local environment with OpenCV installed to see the visual output.")
