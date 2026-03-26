
from ultralytics import YOLO
import cv2
import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans


def classify_rows(boxes):
    """
    boxes: list of [x1,y1,x2,y2,conf,cls]
    returns: row1_idx, row2_idx, sure(boolean)
    """

    N = len(boxes)
    if N == 0:
        return [], [], False

    # Compute y centers
    y_centers = np.array([(b[1] + b[3]) / 2 for b in boxes])
    heights = np.array([b[3] - b[1] for b in boxes])
    median_height = np.median(heights)

    spread = y_centers.max() - y_centers.min()

    # Case 1 — too little vertical spread → 1 row
    if spread < median_height * 0.5:
        sorted_idx = np.argsort([b[0] for b in boxes])
        return sorted_idx.tolist(), [], False

    # Try KMeans → 2 clusters
    kmeans = KMeans(n_clusters=2, n_init='auto').fit(y_centers.reshape(-1, 1))
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_.flatten()

    # Case 2 — centers too close → treat as 1 row
    if abs(centers[0] - centers[1]) < median_height * 0.4:
        sorted_idx = np.argsort([b[0] for b in boxes])
        return sorted_idx.tolist(), [], False

    # Identify top and bottom rows
    top_id = np.argmin(centers)
    bottom_id = np.argmax(centers)

    row1 = [i for i, lab in enumerate(labels) if lab == top_id]
    row2 = [i for i, lab in enumerate(labels) if lab == bottom_id]

    # Step: Correct misclassified boxes by line boundaries
    def correct_row(row, other_row):

        if len(row) < 2:
            return row, other_row  # cannot fit a line

        # Fit line to current row
        Y_1 = np.array([[(boxes[i][1] + boxes[i][3]) / 2] for i in row])
        X_1 = np.array([[(boxes[i][0] + boxes[i][2]) / 2] for i in row])

        Y_2 = np.array([[(boxes[i][1] + boxes[i][3]) / 2] for i in other_row])
        X_2 = np.array([[(boxes[i][0] + boxes[i][2]) / 2] for i in other_row])

        model1 = LinearRegression().fit(X_1, Y_1)
        model2 = LinearRegression().fit(X_2, Y_2)

        slope1 = model1.coef_[0][0]
        intercept1 = model1.intercept_[0]

        slope2 = model2.coef_[0][0]
        intercept2 = model2.intercept_[0]

        A1, B1, C1 = -slope1, 1, -intercept1
        A2, B2, C2 = -slope2, 1, -intercept2

        all_indices = np.array(row + other_row)

        X_all = np.concatenate([X_1, X_2]).flatten()
        Y_all = np.concatenate([Y_1, Y_2]).flatten()

        dist1 = np.abs(A1 * X_all + B1 * Y_all + C1) / np.sqrt(A1**2 + B1**2)
        dist2 = np.abs(A2 * X_all + B2 * Y_all + C2) / np.sqrt(A2**2 + B2**2)

        closer_to_line1 = dist1 < dist2

        row1 = all_indices[closer_to_line1].tolist()
        row2 = all_indices[~closer_to_line1].tolist()

        return row1, row2

    row1, row2 = correct_row(row1, row2)

    # Sort each row left to right
    row1 = sorted(row1, key=lambda idx: boxes[idx][0])
    row2 = sorted(row2, key=lambda idx: boxes[idx][0])

    sure = (len(row1) >= 2 and len(row2) >= 2)
    return row1, row2, sure


def compute_row_angle(boxes_data, row_indices):
    """
    boxes_data: list of bounding box rows [x1, y1, x2, y2, conf, cls]
    row_indices: list of indices (ints) indicating which boxes belong to the row

    Returns: angle in degrees (float)
    """

    # Extract center points of the chosen row
    centers = []
    for i in row_indices:
        x1, y1, x2, y2 = boxes_data[i][:4]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append((cx, cy))

    # Sort by x position (left → right)
    centers = sorted(centers, key=lambda p: p[0])

    # If fewer than 2 points, angle is zero
    if len(centers) < 2:
        return 0.0

    # Fit line (least squares)
    xs = np.array([c[0] for c in centers])
    ys = np.array([c[1] for c in centers])

    a, b = np.polyfit(xs, ys, 1)  # slope a

    # Convert slope → angle
    angle_rad = math.atan(a)
    angle_deg = math.degrees(angle_rad)

    return angle_deg
     

def rotate_image(image, angle):
    """
    Rotates an image without cropping.
    angle: degrees (positive = counter-clockwise)
    """

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding dimensions
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take new size into account
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform rotation
    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255,255,255))

    return rotated


def draw_boxes(img, boxes, row1_idx, row2_idx):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])

        if i in row1_idx:      # First line → red
            color = (0, 0, 255)
        elif i in row2_idx:    # Second line → green
            color = (0, 255, 0)
        else:
            color = (255, 255, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img


def find_rows(results, image_height):
  boxes = results[0].boxes.data.tolist()
  names = results[0].names

  coords = []
  for box in boxes:
    x_min, y_min, x_max, y_max, conf, class_id = box
    coords.append({
        'x_min': x_min,
        'y_min': y_min,
        'x_max': x_max,
        'y_max': y_max,
        'conf': conf,
        'class_id': int(class_id),
        'label': names[int(class_id)]
    })

  y_centers = np.array([(b['y_min'] + b['y_max']) / 2 for b in coords]).reshape(-1, 1)

  kmeans = KMeans(n_clusters=2, random_state=0).fit(y_centers)
  labels = kmeans.labels_
  centroids = kmeans.cluster_centers_

  row1 = []
  row2 = []

  for label, det in zip(labels, coords):
    if label == 0:
       row1.append(det)
    else:
      row2.append(det)

  # Check if two clusters are REAL
  # Step 1. Compute horizontal centers of each cluster
  x_center_cluster0 = float(centroids[0,0])
  x_center_cluster1 = float(centroids[1,0])
  horizontal_diff = abs(x_center_cluster0 - x_center_cluster1)

  # Step 2. Compute mean height of ALL boxes
  heights = [(b['y_max'] - b['y_min']) for b in coords]
  mean_height = np.mean(heights)

  # Step 3. Decision rule
  if horizontal_diff < (1/2) * mean_height:
      # They are technically one row. so merge.
      row = sorted(coords, key=lambda b: b['x_min'])

      # Detecting whether this one row is the first or second
      return [None, row] if mean_height < (image_height / 2) else [row, None]


  # Otherwise return two clusters
  row1 = sorted(row1, key=lambda b: b['x_min'])
  row2 = sorted(row2, key=lambda b: b['x_min'])

  if np.mean([d['y_min'] for d in row1]) > np.mean([d['y_min'] for d in row2]):
     row1, row2 = row2, row1

  return [row1, row2]


def check_lateral_2(boxes, plate_width):
    W = plate_width
    delta = 0.05 * W

    left = []
    right = []
    center = []
    mid = W / 2

    for box in boxes:
        x_min, x_max = box['x_min'], box['x_max']
        x_c = (x_min + x_max) / 2

        if x_c < mid - delta:
            left.append(box)
        elif x_c > mid + delta:
            right.append(box)
        else:
            center.append(box)

    return len(left), len(right)


def check_lateral_1(boxes, plate_width):
    W = plate_width
    delta = 0.05 * W

    left = []
    right = []
    mid = W / 2

    for box in boxes:
        x_min, x_max = box['x_min'], box['x_max']
        x_c = (x_min + x_max) / 2

        if x_c < mid - delta:
            left.append(box)
        elif x_c > mid + delta:
            right.append(box)

    return len(left), len(right)


def miss_detector(yolo_outputs,
                      num_digits,
                      plate_width = 0,
                      conf_threshold = 0.80,
                      max_gap_ratio = 1.5):

    # Step 1: Filter out low-confidence detections
    # filtered_digits = [d for d in yolo_outputs if d['conf'] >= conf_threshold]
    filtered_digits = [
      {**d, 'label': d['label'] if d['conf'] >= conf_threshold else "?"}
      for d in yolo_outputs
    ]

    # Step 2: Sort digits left-to-right using the 'x_min' coordinate
    filtered_digits.sort(key=lambda d: d['x_min'])

    # Step 3: Compute average digit width for gap-based estimation
    widths = [d['x_max'] - d['x_min'] for d in filtered_digits]
    avg_width = sum(widths) / len(widths)

    # Step 4: Initialize plate representation with '?'
    plate = ['?'] * num_digits

    # Step 5: Estimate positions based on relative horizontal placement
    positions_filled = 0
    last_x_max = None

    for i, digit in enumerate(filtered_digits):
        if last_x_max is None:
            # First digit goes into the first empty slot
            plate[positions_filled] = digit['label']
            positions_filled += 1
        else:
            # Compute gap between current digit and previous digit
            gap = digit['x_min'] - last_x_max
            # Estimate how many missing digits fit in the gap
            missing_slots = round(gap / avg_width) - 1
            missing_slots = max(0, missing_slots)

            # Fill in missing slots
            for _ in range(missing_slots):
                if positions_filled < num_digits:
                    positions_filled += 1

            # Insert current digit
            if positions_filled < num_digits:
                plate[positions_filled] = digit['label']
                positions_filled += 1

        # Update last_x_max for next iteration
        last_x_max = digit['x_max']

    # Step 6: Recheck whether the missing numbers are on the left or the right.
    # for the first row
    if num_digits == 3:
        missing_last_one = True if plate[-1] == '?' else False
        missing_last_two = True if plate[-2:] == ['?', '?'] else False

        if missing_last_two or missing_last_one:
            left, right = check_lateral_1(filtered_digits, plate_width)

            if missing_last_two:
                if left == 0:
                    plate = plate[-2:] + plate[:-2]

            elif missing_last_one:
                if left == 0:
                    plate = plate[-1:] + plate[:-1]

    # for the second row
    elif num_digits == 5:
        missing_last_one = True if plate[-1] == '?' else False
        missing_last_two = True if plate[-2:] == ['?', '?'] else False

        if missing_last_two or missing_last_one:
            left, right = check_lateral_2(filtered_digits, plate_width)

            if missing_last_two:
                if left == 1:
                    plate = plate[-1:] + plate[:-1]
                elif left == 0:
                    plate = plate[-2:] + plate[:-2]

            elif missing_last_one:
                if (left == 0) or  (left == 1):
                    plate = plate[-1:] + plate[:-1]


    # Step 7: Return the plate as a string
    return ''.join(plate)


def get_yolo_ocr_result_motor(plate_img, motor_ocr_model, conf_thresh):

    yolo_res = motor_ocr_model.predict(plate_img, agnostic_nms=True)
    yolo_res = yolo_res[0]
    boxes_data = yolo_res.boxes.data.tolist()

    row1, row2, sure = classify_rows(boxes_data)

    # Step 2: rotation image
    # Pick the more reliable row (more boxes)
    if len(row1) >= len(row2):
        row_for_angle = row1
    else:
        row_for_angle = row2

    # Do not rotate it (if Yolo can't find more than 2 boxes)
    if len(row_for_angle) < 2:
        print("Cannot estimate rotation — too few boxes")
        return ""
    else:
        angle = compute_row_angle(boxes_data, row_for_angle)
        print("Rotation angle:", angle)

    # Do not try Yolo twice (Step3) if do not rotate an image
    # TODO: find the proper angle
    if angle == 0:
        rotated_img = plate_img
    else:
        rotated_img = rotate_image(plate_img, angle)

        # Step 3: try yolo again for rotated image
        yolo_res = motor_ocr_model.predict(rotated_img, agnostic_nms=True)

    # Step 4: classify rows
    plate_height, plate_width = rotated_img.shape[:2]
    row1, row2 = find_rows(yolo_res, plate_height)

    # Step 5: miss detector
    if row1 is not None and row2 is not None:
      pred_text_1 = miss_detector(row1, 3, plate_width)
      pred_text_2 = miss_detector(row2, 5, plate_width)
      pred_text = pred_text_1 + "-" + pred_text_2
    #   x_axis = round(plate_width / 2)
    #   cv2.line(rotated_img, (x_axis, 0), (x_axis, 300), color=(0, 0, 255), thickness=2)
    elif row1 is not None:
      pred_text = miss_detector(row1, 3, plate_width)
      pred_text = pred_text + "-?????" 
    elif row2 is not None:
      pred_text = miss_detector(row2, 5, plate_width)
      pred_text = "???-" + pred_text
    
    return pred_text