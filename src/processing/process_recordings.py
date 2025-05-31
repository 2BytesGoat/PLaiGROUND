# %%
import cv2
import json
import numpy as np
# %%
with open("../../data/recording.json", 'r') as f:
    data = json.load(f)
# %%
for recording_session in data:
    break
# %%
observations, observations_2d, actions = recording_session

# %%
def decode_frame(frame_as_string):
    frame_as_bytes = np.frombuffer(bytes.fromhex(frame_as_string), dtype=np.uint8)
    frame_width = int((len(frame_as_bytes) // 3) ** 0.5)
    return frame_as_bytes.reshape(frame_width, frame_width, 3)

frames = []
for frame_as_string in observations_2d:
    frame = decode_frame(frame_as_string)
    frames.append(frame)

# %%
# convert frames to a video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frames[0].shape[1], frames[0].shape[0]))
for frame in frames:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out.write(frame)
out.release()

# %%
import matplotlib.pyplot as plt
plt.imshow(frames[-10])
# %%