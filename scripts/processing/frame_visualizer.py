import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


class FrameVisualizer:
    def __init__(self, extra_features: int = 8):
        self.extra_features = extra_features
        self.sensor_labels = [
            "dir_x",
            "dir_y",
            "vel_x",
            "vel_y",
            "on_floor",
            "on_wall",
            "perc_to_peak",
            "has_powerup",
        ]
        self.class_names = {
            0: "Empty",
            1: "Wall",
            2: "Player",
            3: "Spikes",
            4: "Exit",
            5: "Reset Block",
            6: "Bounce Pad",
            7: "Ice",
            8: "Dissolve Block",
            9: "Double Jump (Powerup)",
            10: "Stomp (Powerup)",
            11: "Dash (Powerup)",
            12: "Grapple (Powerup)",
        }
        self.default_color = "#f8f9fa"
        self.class_colors = {
            0: "#f8f9fa",   # Empty
            1: "#2f9e44",   # Wall
            2: "#f08c00",   # Player
            3: "#e03131",   # Spikes
            4: "#1971c2",   # Exit
            5: "#6741d9",   # Reset Block
            6: "#f76707",   # Bounce Pad
            7: "#74c0fc",   # Ice
            8: "#868e96",   # Dissolve Block
            9: "#ffd43b",   # Double Jump
            10: "#ff922b",  # Stomp
            11: "#ff6b6b",  # Dash
            12: "#20c997",  # Grapple
        }


    def parse_observation(self, observation) -> dict:
        """Parse a raw observation vector into grid + sensor components."""
        state = np.asarray(observation, dtype=float).tolist()
        grid_flat = state[:-self.extra_features]
        extras = np.asarray(state[-self.extra_features:], dtype=float)

        pixel_count = len(grid_flat)
        side = int(np.sqrt(pixel_count))
        if side * side != pixel_count:
            raise ValueError(
                f"Cannot reshape grid of length {pixel_count} into a square grid. "
                "Check extra_features or provide explicit grid dimensions."
            )

        grid = np.asarray(grid_flat, dtype=int).reshape(side, side)
        unique_ids = sorted(np.unique(grid).tolist())

        return {
            "grid": grid,
            "side": side,
            "extras": extras,
            "unique_ids": unique_ids,
            "sensor_values": dict(zip(self.sensor_labels, extras.tolist())),
        }


    def extract_frame_info(self, frame: dict) -> dict:
        """Backward-compatible wrapper for recorded frame dictionaries."""
        if "state" not in frame:
            raise KeyError("Frame is missing 'state'.")
        frame_info = self.parse_observation(frame["state"])
        frame_info["action"] = int(frame.get("action", 0))
        return frame_info


    def format_observation_text(self, observation, include_grid: bool = False) -> str:
        """Create a readable text summary for intros/debugging."""
        frame_info = self.parse_observation(observation)
        sensor_values = frame_info["sensor_values"]
        sensor_lines = [
            f"- {label}: {sensor_values[label]:.3f}" for label in self.sensor_labels
        ]

        sections = [
            "Observation summary",
            f"- grid_size: {frame_info['side']}x{frame_info['side']}",
            f"- classes_seen: {frame_info['unique_ids']}",
            "Sensors:",
            *sensor_lines,
        ]

        if include_grid:
            sections.append("Grid:")
            sections.append(np.array2string(frame_info["grid"], separator=", "))

        return "\n".join(sections)

    def plot_frame_info(
        self,
        frame_info: dict,
        session_id: int = 0,
        frame_index: int = 0,
        action: int = 0,
    ) -> None:
        grid = frame_info["grid"]
        side = frame_info["side"]
        unique_ids = frame_info["unique_ids"]
        extras = frame_info["extras"]

        id_to_idx = {cid: i for i, cid in enumerate(unique_ids)}
        indexed_grid = np.vectorize(lambda v: id_to_idx[int(v)])(grid)
        colors = [self.class_colors.get(cid, self.default_color) for cid in unique_ids]
        cmap = ListedColormap(colors)

        legend_handles = [
            Patch(
                color=self.class_colors.get(cid, self.default_color),
                label=f"{cid}: {self.class_names.get(cid, f'Class {cid}')}",
            )
            for cid in unique_ids
        ]

        fig, (ax_grid, ax_sensors) = plt.subplots(
            2, 1, figsize=(8, 9), gridspec_kw={"height_ratios": [3, 1.6]}
        )

        ax_grid.imshow(indexed_grid, cmap=cmap, interpolation="nearest")
        ax_grid.set_title(
            f"Session {session_id} | Frame {frame_index} | "
            f"Action {frame_info.get('action', action)}"
        )
        # Keep major ticks at cell centers for labels.
        ax_grid.set_xticks(range(side))
        ax_grid.set_yticks(range(side))

        # Shift grid lines by +0.5 cell (right/down) as requested.
        shifted_lines = np.arange(0.5, side + 0.5, 1.0)
        ax_grid.set_xticks(shifted_lines, minor=True)
        ax_grid.set_yticks(shifted_lines, minor=True)
        ax_grid.grid(which="minor", color="black", linewidth=0.5, alpha=0.35)
        ax_grid.tick_params(which="minor", bottom=False, left=False)
        ax_grid.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left")

        x = np.arange(len(self.sensor_labels))
        bar_colors = ["#4c6ef5" if v >= 0 else "#e03131" for v in extras]
        ax_sensors.bar(x, extras, color=bar_colors, alpha=0.9)
        ax_sensors.axhline(0, color="black", linewidth=1)
        ax_sensors.set_xticks(x)
        ax_sensors.set_xticklabels(self.sensor_labels, rotation=25, ha="right")
        ax_sensors.set_ylabel("value")
        ax_sensors.set_title("Extra sensors")

        plt.tight_layout()
        plt.show()

    def plot_observation(
        self, observation, session_id: int = 0, frame_index: int = 0, action: int = 0
    ) -> None:
        frame_info = self.parse_observation(observation)
        frame_info["action"] = action
        self.plot_frame_info(
            frame_info=frame_info,
            session_id=session_id,
            frame_index=frame_index,
            action=action,
        )
