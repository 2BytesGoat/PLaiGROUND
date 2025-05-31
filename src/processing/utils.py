import json
from pathlib import Path


def sample_recording_session(recording_path):
    recording_path = Path(recording_path)
    with open(recording_path, 'r') as f:
        data = json.load(f)

    if len(data) == 0:
        raise ValueError("No data found in recording")
    
    observations, observations_2d, actions = data[0]
    
    sample_recording_session = [
        observations[-100:],
        observations_2d[-100:],
        actions[-100:],
    ]

    return sample_recording_session

if __name__ == "__main__":
    sample_recording_session("data/recording.json")