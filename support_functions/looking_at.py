import math

# calculate the Euclidean distance between a gaze point and the center of a bounding box
def distance_to_bbox_center(gaze_point, bbox):
    """
    Compute Euclidean distance between the gaze point and the center of a bounding box.
    
    :param gaze_point: Tuple (x, y) for the gaze point in image coordinates.
    :param bbox: Tuple (x1, y1, x2, y2) for the bounding box.
    :return: Distance in pixels.
    """
    x1, y1, x2, y2 = bbox
    bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    distance = math.hypot(gaze_point[0] - bbox_center[0], gaze_point[1] - bbox_center[1])
    return distance

# Determine which object the user is looking at based on gaze point and detection bounding boxes
def get_gazed_object(gaze_point, detections, threshold=100):
    """
    Determine which object the user is looking at based on proximity to detection bounding boxes.
    
    :param gaze_point: Tuple (x, y) in image coordinates.
    :param detections: List of detection dicts (each with 'class_name' and 'bbox').
    :param threshold: Max distance in pixels for a gaze to be considered on an object.
    :return: Name of the object being looked at or None.
    """
    min_distance = float("inf")
    closest_object = None

    for detection in detections:
        class_name = detection["class_name"]
        if class_name == "Hand":
            continue  # Skip hand detection

        bbox = detection["bbox"]
        distance = distance_to_bbox_center(gaze_point, bbox)

        if distance < min_distance and distance <= threshold:
            min_distance = distance
            closest_object = class_name

    return closest_object

