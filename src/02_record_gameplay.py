import os
import time

import numpy as np
from pynput import keyboard

from ml_forge.game.recording import build_frame_record, make_session_file, save_frames_to_disk
from utils import setup_environment

SAVE_FOLDER = "data/"
SAVE_FILE_NAME = "recorded_session"
os.makedirs(SAVE_FOLDER, exist_ok=True)

def make_session_file_path() -> str:
    return make_session_file(save_folder=SAVE_FOLDER, file_stem=SAVE_FILE_NAME)

# Global state to communicate between the listener thread and main loop
class InputState:
    def __init__(self):
        self.jump = False
        self.discard = False
        self.quit = False

input_state = InputState()

def on_press(key):
    try:
        if key == keyboard.Key.space:
            input_state.jump = True
        elif hasattr(key, 'char'):
            if key.char == 'r': input_state.discard = True
        elif key == keyboard.Key.esc:
            input_state.quit = True
    except AttributeError:
        pass

def on_release(key):
    if key == keyboard.Key.space:
        input_state.jump = False

def save_to_disk(buffer, filename):
    written = save_frames_to_disk(buffer, filename)
    if written:
        print(f"--- Saved {written} frames to {filename} ---")

def main():
    env = setup_environment()
    obs = env.reset()
    nb_agents = len(obs["obs"])

    episode_buffer = []
    fps_limit = 1/120
    session_id = 0
    session_file = make_session_file_path()

    # Start the non-blocking listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("Controls: SPACE=Jump | R=Discard | ESC=Quit")

    try:
        while not input_state.quit:
            loop_start = time.perf_counter()

            # 1. Reset/Save Logic (Checked once per loop)
            if input_state.discard:
                print("Discarding buffer... starting a new session")
                episode_buffer = []
                # start a new session id (do not save)
                obs = env.reset()
                input_state.discard = False # Reset flag
                time.sleep(0.1)
                continue

            # 2. State & Action, then Step Environment
            observation = obs["obs"][0]
            action = 1 if input_state.jump else 0

            actions = np.full((nb_agents, 1), action, dtype=np.int64)
            obs, reward, done, info = env.step(actions)

            episode_buffer.append(
                build_frame_record(observation, action, reward, done, session_id)
            )

            # 4. Auto-Reset Logic
            if any(done):
                save_to_disk(episode_buffer, session_file)
                session_id += 1

                # Reset for next episode / start new session
                episode_buffer = []
                obs = env.reset()
                input_state.discard = False
                continue

            # 5. Timing
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, fps_limit - elapsed))

    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        # Save any remaining frames from the current session
        if episode_buffer:
            save_to_disk(episode_buffer, session_file)
        env.close()
        print("Environment closed. Saved remaining buffer if any.")

if __name__ == "__main__":
    main()