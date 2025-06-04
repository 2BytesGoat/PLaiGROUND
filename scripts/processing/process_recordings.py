import json
from pathlib import Path

import numpy as np


def decode_frame(frame_as_string):
    frame_as_bytes = np.frombuffer(bytes.fromhex(frame_as_string), dtype=np.uint8)
    frame_width = int((len(frame_as_bytes) // 3) ** 0.5)
    return frame_as_bytes.reshape(frame_width, frame_width, 3)

def save_video(frames, output_path):
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
    out.release()

def process_recording(recording_path, dump_video=False):
    data = []

    recording_path = Path(recording_path)
    with open(recording_path, 'r') as f:
        data = json.load(f)

    if len(data) == 0:
        raise ValueError("No data found in recording")

    observations, observations_2d, actions = None, None, None
    for idx, recording_session in enumerate(data):
        obs, obs_2d_as_string, act = recording_session
        obs_2d = [decode_frame(frame_as_string) for frame_as_string in obs_2d_as_string]

        observations = np.concatenate([observations, obs]) if observations is not None else obs
        observations_2d = np.concatenate([observations_2d, obs_2d]) if observations_2d is not None else obs_2d
        actions = np.concatenate([actions, act]) if actions is not None else act

        if dump_video:
            output_dir = recording_path.parent / "videos"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{recording_path.stem}_{idx:03d}.mp4"
            save_video(obs_2d, output_path)

    return observations, observations_2d, actions

if __name__ == "__main__":
    process_recording("data/recording.json", dump_video=True)
