# real-time-object-counting
This project provides a Python implementation for counting objects moving across a predefined virtual line in a video stream.

Technology Stack:
Python 3.x
OpenCV (cv2): For video frame manipulation and visualization.
NumPy: For efficient array operations.
SciPy (scipy.spatial.distance): Used specifically for the cdist function, which calculates the distance matrix for matching new detections to existing tracks.
collections.deque: Used in TrackableObject to efficiently store a history of recent centroids for movement analysis.
