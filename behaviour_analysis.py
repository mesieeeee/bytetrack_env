import numpy as np
import math
from scipy.spatial import ConvexHull

# Utility functions
def euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def angle(a, b, c):
    ab = np.array(a) - np.array(b)
    cb = np.array(c) - np.array(b)
    cos = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    return math.degrees(math.acos(np.clip(cos, -1, 1)))

def center_of_mass(points):
    pts = np.array(points)
    return pts.mean(axis=0) if len(pts) else np.array([0, 0])

def valid_points(points):
    """Check if all given keypoints are valid (not None, no NaNs)."""
    return all(
        p is not None and len(p) >= 2 and not np.isnan(p[0]) and not np.isnan(p[1])
        for p in points
    )

# 1. Movement & Posture
def dribbling_stance(kps_seq):
    dist = []
    for f in kps_seq:
        if len(f) < 13:  # safety check for indexing
            dist.append(None)
            continue
        l_wrist, r_wrist, l_hip, r_hip = f[9], f[10], f[11], f[12]
        if valid_points([l_wrist, r_wrist, l_hip, r_hip]):
            hip_mid_y = (l_hip[1] + r_hip[1]) / 2
            wrist_min_y = min(l_wrist[1], r_wrist[1])
            dist.append(hip_mid_y - wrist_min_y)
        else:
            dist.append(None)
    return np.nanmean([d for d in dist if d is not None])

def footwork_dynamics(kps_seq):
    speeds, lateral = [], []
    for f1, f2 in zip(kps_seq, kps_seq[1:]):
        if len(f1) < 17 or len(f2) < 17:
            continue
        a1 = center_of_mass([f1[15], f1[16]])  # left & right foot
        a2 = center_of_mass([f2[15], f2[16]])
        speeds.append(euclidean(a1, a2))
        lateral.append(abs(a2[0] - a1[0]))
    return np.nanmean(speeds), sum(l > 5 for l in lateral)

def fatigue_indicator(kps_seq):
    heights = []
    for f in kps_seq:
        if len(f) < 17:
            continue
        if valid_points([f[0], f[15], f[16]]):
            heights.append(abs(f[0][1] - center_of_mass([f[15], f[16]])[1]))
    if len(heights) > 1:
        return np.polyfit(range(len(heights)), heights, 1)[0]
    return 0

def recovery_balance(kps_seq):
    coms = []
    for f in kps_seq:
        if len(f) >= 13 and valid_points([f[5], f[6], f[11], f[12]]):
            coms.append(center_of_mass([f[5], f[6], f[11], f[12]]))
    disps = [euclidean(c1, c2) for c1, c2 in zip(coms, coms[1:])]
    return np.nanstd(disps), np.nanmax(disps) if disps else (0, 0)

# 2. Ball Handling
def dribble_control(kps_seq, ball_seq):
    wrist_y = [k[9][1] for k in kps_seq if len(k) > 11 and valid_points([k[9], k[11]])]
    wrist_y = np.array(wrist_y)
    return wrist_y.std(), np.mean(np.diff(wrist_y) != 0) if len(wrist_y) > 1 else (0, 0)

def crossover_efficiency(kps_seq):
    switches = 0
    for w1, w2 in zip(kps_seq[:-1], kps_seq[1:]):
        if len(w1) > 10 and len(w2) > 10 and valid_points([w1[9], w1[10], w2[9], w2[10]]):
            cond1 = w1[9][1] < w1[10][1]
            cond2 = w2[9][1] < w2[10][1]
            if cond1 != cond2:
                switches += 1
    return switches / len(kps_seq) if kps_seq else 0

def directional_smoothness(kps_seq):
    angles = []
    for f1, f2 in zip(kps_seq[:-1], kps_seq[1:]):
        if len(f1) > 12 and len(f2) > 12:
            angles.append(angle(f1[11], f1[12], f2[12]))
    return (np.nanmean(np.diff(angles)), np.nanstd(angles)) if angles else (0, 0)

# 3. Shooting Mechanics
def release_timing(kps_seq, shot_frames):
    return [j - i for i, j in shot_frames]

def wrist_elbow_alignment(kps_seq, shot_idx):
    f = kps_seq[shot_idx]
    return angle(f[5], f[7], f[9]) if len(f) > 9 else 0

def follow_through(kps_seq, shot_idx):
    if shot_idx + 1 < len(kps_seq) and len(kps_seq[shot_idx]) > 9:
        return euclidean(kps_seq[shot_idx][9], kps_seq[shot_idx+1][9])
    return 0

def arc_consistency(kps_seq, shot_indices):
    arc_heights = [kps_seq[i][9][1] for i in shot_indices if len(kps_seq[i]) > 9]
    return np.nanstd(arc_heights) if arc_heights else 0

# 4. Defensive Positioning
def closeout_stance(kps_seq):
    angles = [angle(f[15], f[13], f[11]) for f in kps_seq if len(f) > 15]
    return np.nanmean(angles) if angles else 0

def lateral_quickness(kps_seq):
    speeds = []
    for f1, f2 in zip(kps_seq, kps_seq[1:]):
        if len(f1) > 16 and len(f2) > 16:
            a1 = center_of_mass([f1[15], f1[16]])
            a2 = center_of_mass([f2[15], f2[16]])
            speeds.append(abs(a2[0] - a1[0]))
    return np.nanmean(speeds) if speeds else 0

# 5. Passing Decisions
def suboptimal_pass(passer, receiver, defender):
    line = np.array(receiver) - np.array(passer)
    perp = abs(np.cross(line, np.array(defender) - np.array(passer))) / (np.linalg.norm(line) + 1e-6)
    return perp < 20

def assist_potential(off_players, def_players):
    if len(off_players) < 3:
        return 0
    hull = ConvexHull([p for p in off_players])
    area = hull.volume
    return area / (len(def_players) + 1)

# 6. Off-Ball Movement
def cut_timing(speed_seq):
    return int(np.argmax(speed_seq)) if speed_seq else 0

def floor_spacing(off_players):
    pts = np.array(off_players)
    if len(pts) < 3:
        return 0
    return ConvexHull(pts).volume / len(pts)

# 7. Transition Play
def reaction_time(turnover_frame, movement_seq):
    return int(np.argmax([1 if m > 5 else 0 for m in movement_seq[turnover_frame:]]))

def fast_break_position(off_position, def_positions):
    if not def_positions:
        return 0
    dists = [euclidean(off_position, d) for d in def_positions]
    return min(dists)
