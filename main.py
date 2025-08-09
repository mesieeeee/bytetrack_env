import os
import csv
import numpy as np
import cv2
from trackers import PlayerTracker, BallTracker
from utils import read_video, save_video
from drawers import PlayerTracksDrawer, BallTracksDrawer, PoseDrawer
from pose_estimator import PoseEstimator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')  # Ensure non-interactive backend for video rendering
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# Import behaviour analysis functions
from behaviour_analysis import (
    dribbling_stance, footwork_dynamics, fatigue_indicator, recovery_balance,
    dribble_control, crossover_efficiency, directional_smoothness,
    closeout_stance, lateral_quickness
)

# --- Utility to ensure YOLOv8 keypoints are always length 17 ---
def safe_keypoints(keypoints, target_len=17):
    """Pad or trim keypoints to ensure fixed length."""
    if keypoints is None:
        return [(0, 0)] * target_len
    if len(keypoints) < target_len:
        keypoints = keypoints + [(0, 0)] * (target_len - len(keypoints))
    elif len(keypoints) > target_len:
        keypoints = keypoints[:target_len]
    return keypoints

# --- Generate 3D Visualization Video ---
def generate_3d_visualization(player_kps_sequences, analysis_results, output_path="output_videos/3d_analysis.avi"):
    """
    Generates a 3D visualization video of player keypoints over time
    with analysis insights and corrective feedback.
    """
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = 10
    size = (640, 480)
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    for player_id, kps_seq in player_kps_sequences.items():
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111, projection="3d")

        xs, ys, zs = [], [], []
        for frame_idx, keypoints in enumerate(kps_seq):
            if isinstance(keypoints, list) and all(len(kp) == 2 for kp in keypoints):
                xs.extend([x for x, y in keypoints])
                ys.extend([y for x, y in keypoints])
                zs.extend([frame_idx] * len(keypoints))

        ax.scatter(xs, ys, zs, c=zs, cmap="viridis", marker="o")
        ax.set_title(f"Player {player_id} - 3D Movement Analysis")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Frame")

        # Add corrective feedback text
        feedback = next((res for res in analysis_results if res["player_id"] == player_id), None)
        if feedback:
            ax.text2D(0.05, 0.95, f"Avg Speed: {feedback['avg_speed']:.2f}", transform=ax.transAxes)
            ax.text2D(0.05, 0.90, f"Fatigue: {feedback['fatigue_slope']:.2f}", transform=ax.transAxes)

        # Convert matplotlib fig to OpenCV image
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        img = cv2.resize(img, size)
        out.write(img)
        plt.close(fig)

    out.release()
    print(f"✅ 3D visualization video saved to: {output_path}")


# --- Add Corrective Feedback Overlay ---
def add_feedback_overlay(video_path, analysis_results, output_path="output_videos/final_with_feedback1.avi"):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20, (int(cap.get(3)), int(cap.get(4))))

    feedback_dict = {}
    for res in analysis_results:
        fb = []
        if res["dribbling_stance"] < 0.5:
            fb.append("Widen stance")
        if res["avg_speed"] < 1.0:
            fb.append("Increase movement speed")
        if res["lateral_quickness"] < 0.5:
            fb.append("Improve lateral quickness")
        feedback_dict[res["player_id"]] = ", ".join(fb) if fb else "Good form"

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        y_offset = 30
        for pid, fb_text in feedback_dict.items():
            cv2.putText(frame, f"Player {pid}: {fb_text}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            y_offset += 30
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Final feedback video saved to: {output_path}")

def main():
    # ----------------- 1. Load Video -----------------
    video_path = r'C:\Users\madam\OneDrive\Desktop\basket-ball\bytetrack_env\test\Video-2.mp4'
    video_frames = read_video(video_path)

    # ----------------- 2. Initialize Trackers -----------------
    player_tracker = PlayerTracker(r'C:\Users\madam\OneDrive\Desktop\basket-ball\bytetrack_env\models\player_detector.pt')
    ball_tracker = BallTracker(r'C:\Users\madam\OneDrive\Desktop\basket-ball\bytetrack_env\models\ball_detector_model.pt')

    # ----------------- 3. Get Tracks -----------------
    player_tracks = player_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/player_track_stubs.pkl")
    ball_tracks = ball_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/ball_track_stubs.pkl')
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    # ----------------- 4. Initialize Drawing & Pose -----------------
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    pose_drawer = PoseDrawer()
    pose_estimator = PoseEstimator("models/yolov8n-pose.pt")

    keypoints_data = []
    player_kps_sequences = {}
    ball_positions_seq = []

    # ----------------- 5. Process Each Frame -----------------
    for frame_num, frame in enumerate(video_frames):
        player_dict = player_tracks[frame_num]
        for track_id, player in player_dict.items():
            bbox = player["bbox"]
            keypoints = pose_estimator.get_keypoints(frame, bbox)
            keypoints = safe_keypoints(keypoints)
            player["keypoints"] = keypoints

            for kp_index, (x, y) in enumerate(keypoints):
                keypoints_data.append([frame_num, track_id, kp_index, x, y])

            if track_id not in player_kps_sequences:
                player_kps_sequences[track_id] = []
            player_kps_sequences[track_id].append(keypoints)

            frame = pose_drawer.draw(frame, keypoints)

        frame = player_tracks_drawer.draw_single(frame, player_dict)

        if frame_num < len(ball_tracks):
            ball_positions_seq.append(ball_tracks[frame_num])
            frame = ball_tracks_drawer.draw_single(frame, ball_tracks[frame_num])
        else:
            ball_positions_seq.append(None)

        video_frames[frame_num] = frame

    # ----------------- 6. Save Annotated Video -----------------
    output_video_path = "output_videos/output_video2.avi"
    save_video(video_frames, output_video_path)

    # ----------------- 7. Save Raw Keypoints CSV -----------------
    with open("output_videos/player_keypoints2.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "track_id", "keypoint_index", "x", "y"])
        writer.writerows(keypoints_data)

    # ----------------- 8. Run Behaviour Analysis -----------------
    analysis_results = []
    for track_id, kps_seq in player_kps_sequences.items():
        stance = dribbling_stance(kps_seq)
        avg_speed, lateral_moves = footwork_dynamics(kps_seq)
        fatigue_slope = fatigue_indicator(kps_seq)
        balance_std, balance_max = recovery_balance(kps_seq)
        dribble_std, dribble_changes = (np.nan, np.nan)
        if any(ball_positions_seq):
            try:
                dribble_std, dribble_changes = dribble_control(kps_seq, ball_positions_seq)
            except Exception:
                pass
        crossover = crossover_efficiency(kps_seq)
        dir_smooth_mean, dir_smooth_std = directional_smoothness(kps_seq)
        closeout = closeout_stance(kps_seq)
        lat_quick = lateral_quickness(kps_seq)

        analysis_results.append({
            "player_id": track_id,
            "dribbling_stance": stance,
            "avg_speed": avg_speed,
            "lateral_moves": lateral_moves,
            "fatigue_slope": fatigue_slope,
            "balance_std": balance_std,
            "balance_max": balance_max,
            "dribble_std": dribble_std,
            "dribble_changes": dribble_changes,
            "crossover_efficiency": crossover,
            "dir_smooth_mean": dir_smooth_mean,
            "dir_smooth_std": dir_smooth_std,
            "closeout_stance": closeout,
            "lateral_quickness": lat_quick
        })

    # ----------------- 9. Save Analysis CSV -----------------
    if analysis_results:
        with open("output_videos/behaviour_analysis2.csv", 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=analysis_results[0].keys())
            writer.writeheader()
            writer.writerows(analysis_results)

    # ----------------- 10. Generate 3D Visualization -----------------
    generate_3d_visualization(player_kps_sequences, analysis_results)

    # ----------------- 11. Add Corrective Feedback Overlay -----------------
    add_feedback_overlay(output_video_path, analysis_results)

if __name__ == "__main__":
    main()

