from .utils import draw_ellipse,draw_traingle

class PlayerTracksDrawer:
    def __init__(self,team_1_color=[255, 245, 238],team_2_color=[128, 0, 0]):
        pass

    def draw(self,video_frames,tracks):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks[frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                frame = draw_ellipse(frame, player["bbox"],(0,0,255), track_id)

            output_video_frames.append(frame)

        return output_video_frames
        
    def draw_single(self, frame, player_dict):
        for track_id, player in player_dict.items():
            frame = draw_ellipse(frame, player["bbox"], (0, 0, 255), track_id)
        return frame
