import os
import json
import time
from datetime import datetime

import numpy as np
from pynput import keyboard

from utils import setup_environment

SAVE_FOLDER = "data/"
SAVE_FILE_NAME = "recorded_session"
os.makedirs(SAVE_FOLDER, exist_ok=True)

def make_session_file() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(SAVE_FOLDER, f"{SAVE_FILE_NAME}_{ts}.jsonl")

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
    if not buffer:
        return
    with open(filename, "a", encoding="utf-8") as f:
        for frame in buffer:
            f.write(json.dumps(frame))
            f.write("\n")
    print(f"--- Saved {len(buffer)} frames to {filename} ---")

def main():
    env = setup_environment()
    obs = env.reset()
    nb_agents = len(obs["obs"])
    
    episode_buffer = []
    fps_limit = 1/120
    session_id = 0
    session_file = make_session_file()

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

            # normalize reward/done for JSON
            try:
                reward_serial = np.asarray(reward).tolist()
            except Exception:
                reward_serial = reward
            try:
                done_serial = list(map(bool, done))
            except Exception:
                done_serial = done

            episode_buffer.append({
                "state": observation.tolist(),
                "action": int(action),
                "reward": reward_serial,
                "done": done_serial,
                "session": session_id
            })
            
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
