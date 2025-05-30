import cv2

import numpy as np

import csv

from ultralytics import YOLO

from karman import ExtendKalmanFilter

# Initialize YOLO model
model = YOLO("../runs/train/exp/weights/best.pt")

# Open video
video = "test6.mp4"

video_capture = cv2.VideoCapture(video)

frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Initialize video writer
out_path = "test6.avi"

fourcc = cv2.VideoWriter.fourcc(*"XVID")

out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

# Initialize tracking variables
trajectories = []  # List of trajectories

ekf_list = []      # List of EKF instances for each trajectory

future_steps = fps * 5  # Predict 5 seconds into the future

measurement = []

# Initialize CSV file for storing predicted trajectories
csv_file = open("predicted_trajectories.csv", "w", newline="")

csv_writer = csv.writer(csv_file)

csv_writer.writerow(["TargetID", "Frame", "PredictedX", "PredictedY"])  # Write header

frame_count = 0  # Track frame number

while video_capture.isOpened():

    ret, frame = video_capture.read()

    if not ret:

        break

    frame_count += 1  # Increment frame number

    # Detect objects using YOLO
    results = model.predict(frame, conf=0.25)

    current_centers = []

    for result in results:

        boxes = result.boxes.cpu().numpy()

        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = int((x1 + x2) / 2)

            cy = int((y1 + y2) / 2)

            current_centers.append((cx, cy))

            name = model.names[int(box.cls[0])]

            conf = float(box.conf)

            label = f"{name}  {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Associate detections with existing trajectories
    for center in current_centers:

        if not trajectories:

            # Initialize new trajectory and EKF
            trajectories.append([center])

            ekf = ExtendKalmanFilter(4, 2)

            ekf.x = np.array([center[0], center[1], 0, 0])  # Initial state (x, y, vx, vy)

            ekf_list.append(ekf)

        else:

            # Find the closest existing trajectory
            min_distance = float('inf')

            best_index = 0

            for i, trajectory in enumerate(trajectories):

                last_point = trajectory[-1]

                distance = np.sqrt((last_point[0] - center[0]) ** 2 + (last_point[1] - center[1]) ** 2)

                if distance < min_distance:

                    min_distance = distance

                    best_index = i

            if min_distance < 15:

                # Update trajectory and EKF
                trajectories[best_index].append(center)

                vx = (center[0] - trajectories[best_index][-2][0]) * fps

                vy = (center[1] - trajectories[best_index][-2][1]) * fps

                measurement.append([center[0], center[1]])

                ekf_list[best_index].predict(1 / fps)

                ekf_list[best_index].update(measurement[-1])

            else:

                # Initialize new trajectory and EKF
                trajectories.append([center])

                ekf = ExtendKalmanFilter(4, 2)

                ekf.x = np.array([center[0], center[1], 0, 0])  # Initial state

                ekf_list.append(ekf)

    # Predict and draw future trajectory for each object
    for i, ekf in enumerate(ekf_list):

        predicted_trajectory = []

        current_state = ekf.x.copy()  # Start from the current state

        for step in range(future_steps):

            # Predict the next state
            ekf.predict(1 / fps)

            if len(measurement) > 0:

                last_point = measurement[-1]

                ekf.update(last_point)

                predicted_state = ekf.x

                predicted_trajectory.append((int(predicted_state[0]), int(predicted_state[1])))

                # Write predicted trajectory to CSV
                csv_writer.writerow([i + 1, frame_count + step, predicted_state[0], predicted_state[1]])

        # Draw the predicted trajectory
        for j in range(1, len(predicted_trajectory)):

            cv2.line(frame, predicted_trajectory[j - 1], predicted_trajectory[j], (255, 0, 0), 2)  # Blue color for future trajectory

    # Write frame to output video
    out.write(frame)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

# Release resources
video_capture.release()

out.release()

csv_file.close()

cv2.destroyAllWindows()

# Print trajectories
for i, trajectory in enumerate(trajectories):

    print(f"Trajectory {i + 1}: {trajectory}")