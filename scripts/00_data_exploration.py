import json
from collections import defaultdict
from scripts.processing.frame_visualizer import FrameVisualizer


def load_observations_by_session(data_path: str) -> dict[int, list[dict]]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    frames_by_session = defaultdict(list)
    for frame in data:
        frames_by_session[int(frame["session"])].append(frame)
    return dict(frames_by_session)


if __name__ == "__main__":
    data_path = "data/demo.jsonl"
    session_id = 0
    frame_index = 18

    observations_by_session = load_observations_by_session(data_path)
    visualizer = FrameVisualizer()
    frame_info = visualizer.extract_frame_info(observations_by_session[session_id][frame_index])
    visualizer.plot_frame_info(frame_info, session_id, frame_index)

    print("Sensor values:")
    print(frame_info["sensor_values"])
