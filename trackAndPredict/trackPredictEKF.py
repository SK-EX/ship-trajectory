from tarfile import tar_filter

import cv2

import numpy as np

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

future_steps = fps  # Predict 1 second into the future

while video_capture.isOpened():

    ret, frame = video_capture.read()

    if not ret:

        break

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

    index = 0

    future_predictions = []

    for center in current_centers:

        #初始化轨s迹
        if not trajectories:

            # Initialize new trajectory and EKF
            trajectories.append([center])

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

                measurement = np.array([center[0], center[1]])

            else:

                # Initialize new trajectory and EKF
                trajectories.append([center])


    # Predict future trajectory for each object

    for trajectory in trajectories:

        for i in range(1, len(trajectory)):

            cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

ekf_list = [[] for _ in range(len(trajectories))]

future_predictions = [[] for _ in range(len(trajectories))]

for i in range(len(trajectories)):

    ekf = ExtendKalmanFilter(4, 2)

    ekf.x = np.array([trajectories[i][0][0], trajectories[i][0][1], 0, 0])

    for j in range(len(trajectories[i])):

        ekf.predict(1/ fps)

        ekf.update([trajectories[i][j][0], trajectories[i][j][1]])

        ekf_list[i].append(ekf.x[:2])

        state_predict = ekf.x

        future_state = []

        for k in range(future_steps):

            state_predict = ekf.state_transition_function(ekf.x, 1/ fps)

            future_state.append(state_predict[:2])

        future_predictions[i].append(future_state)

ekf_list = np.array(ekf_list)

future_predictions = np.array(future_predictions)

first_dim = np.shape(future_predictions)[0]

second_dim = np.shape(future_predictions)[1]

future_predictions.reshape( [first_dim , second_dim * fps, 2])

print(future_predictions)
# Release resources
video_capture.release()

out.release()

cv2.destroyAllWindows()

print(np.shape(future_predictions))


# Print trajectories
# for i, trajectory in enumerate(trajectories):
#
#     print(f"Trajectory {i + 1}: {trajectory}")

