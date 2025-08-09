The primary script for this project is main.py. This script handles the entire workflow: loading a video, tracking players and the ball, performing pose estimation, and then running a series of behavioural analyses.

To run the full analysis pipeline, simply execute the following command:

python main.py


The script will produce the following outputs in the output_videos/ directory:

output_video2.avi: The original video with player and ball tracks and pose keypoints overlaid.

player_keypoints2.csv: A CSV file containing the raw coordinates for all detected player keypoints per frame.

behaviour_analysis2.csv: A CSV file with a summary of the behavioural analysis for each player, including metrics like average speed, fatigue slope, and dribble control.

3d_analysis.avi: A video visualizing player movement in 3D over time.

final_with_feedback1.avi: The annotated video with corrective feedback text overlaid on the screen.

Note: Before running main.py, you must ensure that your video_path and model paths are correctly configured within the script.
