import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from ml_forge.game.observation import (
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_COLOR,
    SENSOR_LABELS,
    ObservationParser,
)


class FrameVisualizer:
    def __init__(self, extra_features: int = 8):
        self.parser = ObservationParser(extra_features=extra_features)
        self.extra_features = extra_features
        self.sensor_labels = list(SENSOR_LABELS)
        self.class_names = dict(CLASS_NAMES)
        self.default_color = DEFAULT_COLOR
        self.class_colors = dict(CLASS_COLORS)


    def parse_observation(self, observation) -> dict:
        return self.parser.parse_observation(observation)


    def get_class_index(self, class_name: str) -> int:
        """Helper to get class index by name, with error handling."""
        return self.parser.get_class_index(class_name)


    def extract_frame_info(self, frame: dict) -> dict:
        """Backward-compatible wrapper for recorded frame dictionaries."""
        if "state" not in frame:
            raise KeyError("Frame is missing 'state'.")
        frame_info = self.parse_observation(frame["state"])
        frame_info["action"] = int(frame.get("action", 0))
        return frame_info


    def format_observation_text(self, observation, include_grid: bool = False) -> str:
        """Create a readable text summary for intros/debugging."""
        return self.parser.format_observation_text(observation, include_grid=include_grid)

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