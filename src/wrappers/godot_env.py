"""
This script is derived from code originally published in the repository:
https://github.com/edbeeching/godot_rl_agents

All rights to the original work belong to its author(s). In particular,
credit goes to Ed Beeching for creating and maintaining the project.

I do not claim ownership of the original code and am grateful for
Ed Beeching's contributions to the community. This file may contain
minor modifications or adaptations for my specific use case.
"""

import atexit
import json
import os
import pathlib
import socket
import subprocess
import time
from collections import OrderedDict
from sys import platform
from typing import Optional

import numpy as np
from gymnasium import spaces

from wrappers.utils import ActionSpaceProcessor, convert_macos_path


def _normalise_spawn(entry: dict) -> dict:
    """Validate one reset-spawn entry before sending it to Godot."""
    if not isinstance(entry, dict):
        raise ValueError(f"spawn entry must be a dict, got {type(entry)!r}")
    try:
        agent = int(entry["agent"])
        cell = entry["cell"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"spawn entry needs agent and cell: {entry!r}") from exc
    if not isinstance(cell, (list, tuple)) or len(cell) != 2:
        raise ValueError(f"spawn cell must be [x, y], got {cell!r}")
    try:
        facing = int(entry.get("facing", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"spawn facing must be numeric: {entry!r}") from exc
    result = {
        "agent": agent,
        "cell": [int(cell[0]), int(cell[1])],
        "facing": 1 if facing >= 0 else -1,
    }
    state = entry.get("state")
    if isinstance(state, dict):
        # Optional grounded trajectory snapshot. The game validates the cell
        # first, then restores position/velocity/run state after Player.reset().
        result["state"] = state
    return result


class GodotEnv:
    MAJOR_VERSION = "0"  # Versioning for the environment
    MINOR_VERSION = "7"
    DEFAULT_PORT = 11008  # Default port for communication with Godot Game
    DEFAULT_TIMEOUT = 60  # Default socket timeout TODO

    def __init__(
        self,
        env_path: str = None,
        port: int = DEFAULT_PORT,
        show_window: bool = False,
        seed: int = 0,
        framerate: Optional[int] = None,
        action_repeat: Optional[int] = None,
        speedup: Optional[int] = None,
        convert_action_space: bool = False,
        log_path: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        **kwargs,
    ):
        """
        Initialize a new instance of GodotEnv

        Args:
            env_path (str): path to the godot binary environment.
            port (int): Port number for communication.
            show_window (bool): flag to display Godot game window.
            seed (int): seed to initialize the environment.
            framerate (int): the framerate to run the Godot game at.
            action_repeat (int): the number of frames to repeat an action for.
            speedup (int): the factor to speedup game time by.
            convert_action_space (bool): flag to convert action space.
            log_path (str): optional file for the game process stdout/stderr.
        """

        self.proc = None
        self.log_handle = None
        if env_path is not None and env_path != "debug":
            env_path = self._set_platform_suffix(env_path)

            self.check_platform(env_path)
            self._launch_env(
                env_path,
                port,
                show_window,
                framerate,
                seed,
                action_repeat,
                speedup,
                log_path=log_path,
                env=env,
                **kwargs,
            )
        else:
            print("No game binary has been provided, please press PLAY in the Godot editor")

        self.port = port
        self.connection = self._start_server()
        self.num_envs = None
        self.supports_agent_reset = False
        self.supports_spawn_cells = False
        self.supports_state_reset = False
        self.last_reset_agents: list[int] = []
        self.last_reset_spawns: list[dict] = []
        self.info_schema = 1
        self.static_info: dict = {}
        self.last_profile: dict = {}
        self._spawn_cells: list[list[int]] = []
        self._handshake()

        # Action and observation spaces for each in-game agent/env/AIController (used only for multi-agent case with Rllib for now)
        self.action_spaces = []
        self.observation_spaces = []

        self._get_env_info()

        # Single-agent observation space
        self.observation_space = self.observation_spaces[0]

        # sf2 requires a tuple action space
        # Multiple agents' action space(s)
        self.tuple_action_spaces = [
            spaces.Tuple([v for _, v in action_space.items()]) for action_space in self.action_spaces
        ]
        # Single agent action space processor using the action space(s) of the first agent
        self.action_space_processor = ActionSpaceProcessor(self.tuple_action_spaces[0], convert_action_space)

        # For multi-policy envs: The name of each agent's policy set in the env itself (any training_mode
        # AIController instance is treated as an agent)
        self.agent_policy_names

        atexit.register(self._close)

    def _set_platform_suffix(self, env_path: str) -> str:
        """
        Set the platform suffix for the given environment path based on the platform.

        Args:
            env_path (str): The environment path.

        Returns:
            str: The environment path with the platform suffix.
        """
        suffixes = {
            "linux": ".x86_64",
            "linux2": ".x86_64",
            "darwin": ".app",
            "win32": ".exe",
        }
        suffix = suffixes[platform]
        return str(pathlib.Path(env_path).with_suffix(suffix))

    def check_platform(self, filename: str):
        """
        Check the platform and assert the file type

        Args:
            filename (str): Path of the file to check.

        Raises:
            AssertionError: If the file type does not match with the platform or file does not exist.
        """
        if platform == "linux" or platform == "linux2":
            # Linux
            assert (
                pathlib.Path(filename).suffix == ".x86_64"
            ), f"Incorrect file suffix for {filename=} {pathlib.Path(filename).suffix=}. Please provide a .x86_64 file"
        elif platform == "darwin":
            # OSX
            assert (
                pathlib.Path(filename).suffix == ".app"
            ), f"Incorrect file suffix for {filename=} {pathlib.Path(filename).suffix=}. Please provide a .app file"
        elif platform == "win32":
            # Windows...
            assert (
                pathlib.Path(filename).suffix == ".exe"
            ), f"Incorrect file suffix for {filename=} {pathlib.Path(filename).suffix=}. Please provide a .exe file"
        else:
            assert 0, f"unknown filetype {pathlib.Path(filename).suffix}"
        
        assert os.path.exists(filename)

    def from_numpy(self, action, order_ij=False):
        """
        Handles dict to tuple actions

        Args:
            action: The action to be converted.
            order_ij (bool): Order flag.

        Returns:
            list: The converted action.
        """
        result = []

        for agent_idx in range(self.num_envs):
            env_action = {}
            for j, k in enumerate(self.action_spaces[agent_idx].keys()):
                if order_ij is True:
                    v = action[agent_idx][j]
                else:
                    v = action[j][agent_idx]

                if isinstance(v, np.ndarray):
                    env_action[k] = v.tolist()
                else:
                    env_action[k] = int(v)  # cannot serialize int32

            result.append(env_action)
        return result

    def step(self, action, order_ij=False):
        """
        Perform one step in the environment.

        Args:
            action: Action to be taken.
            order_ij (bool): Order flag.

        Returns:
            tuple: Tuple containing observation, reward, done flag, termination flag, and info.
        """
        self.step_send(action, order_ij=order_ij)
        return self.step_recv()

    def step_send(self, action, order_ij=False):
        """
        Send the action to the Godot environment.

        Args:
            action: Action to be sent.
            order_ij (bool): Order flag.
        """
        action = self.action_space_processor.to_original_dist(action)

        message = {
            "type": "action",
            "action": self.from_numpy(action, order_ij=order_ij),
        }
        self._send_as_json(message)

    def step_recv(self):
        """
        Receive the step response from the Godot environment.

        Returns:
            tuple: Tuple containing observation, reward, done flag, termination flag, and info.
        """
        response = self._get_response()
        response["obs"] = self._process_obs(response["obs"])

        # Kept for backward compatibility if the plugin doesn't send info.
        default_info = [{}] * len(response["done"])

        return (
            response["obs"],
            response["reward"],
            np.array(response["done"]).tolist(),
            np.array(response["done"]).tolist(),  # TODO update API to term, trunc
            self._decode_info(response.get("info", default_info)),
        )

    def _process_obs(self, response_obs: dict):
        """
        Process observation data.

        Args:
            response_obs (dict): The response observation to be processed.

        Returns:
            dict: The processed observation data.
        """
        for k in response_obs[0].keys():
            if "2d" in k:
                for sub in response_obs:
                    sub[k] = self._decode_2d_obs_from_string(sub[k], self.observation_space[k].shape)

        return response_obs

    def _apply_reset_info(self, response: dict) -> None:
        """Cache per-episode spawn cells sent only on reset."""
        reset_info = response.get("reset_info")
        if not isinstance(reset_info, dict):
            return
        agents = reset_info.get("agents", [])
        cells = reset_info.get("spawn_cell", [])
        for position, agent in enumerate(agents):
            try:
                index = int(agent)
            except (TypeError, ValueError):
                continue
            if position >= len(cells):
                continue
            cell = cells[position]
            if (
                isinstance(cell, (list, tuple))
                and len(cell) == 2
                and 0 <= index < len(self._spawn_cells)
            ):
                self._spawn_cells[index] = [int(cell[0]), int(cell[1])]

    def _decode_info(self, info):
        """Reconstruct the legacy per-agent info list from the columnar wire
        format. Static fields and spawn cells are merged from env_info/reset
        caches, so no information is dropped."""
        if not isinstance(info, dict) or info.get("schema") != 2:
            return info

        count = len(info.get("run_time", []))
        if len(self._spawn_cells) < count:
            self._spawn_cells.extend(
                [[-1, -1] for _ in range(count - len(self._spawn_cells))]
            )
        static = self.static_info
        decoded: list[dict] = []
        for index in range(count):
            item = {
                "global_position": info["global_position"][index],
                "spawn_cell": self._spawn_cells[index],
                "facing_direction": info["facing_direction"][index],
                "state": info["state"][index],
                "cell": info.get("cell", [[-1, -1]] * count)[index],
                "velocity": info.get("velocity", [[0.0, 0.0]] * count)[index],
                "on_floor": bool(info.get("on_floor", [False] * count)[index]),
                "on_wall": bool(info.get("on_wall", [False] * count)[index]),
                "perc_to_peak": float(info.get("perc_to_peak", [0.0] * count)[index]),
                "powerup_count": int(info.get("powerup_count", [0] * count)[index]),
                "run_active": bool(info.get("run_active", [False] * count)[index]),
                "started_walking": bool(
                    info.get("started_walking", [False] * count)[index]
                ),
                "obs_encoding": static.get("obs_encoding", "legacy"),
                "obs_size": int(static.get("obs_size", 0)),
                "distinct_cells_touched": info["distinct_cells_touched"][index],
                "run_time": info["run_time"][index],
                "best_run_time": info["best_run_time"][index],
                "finished": bool(info["finished"][index]),
                "deaths": int(info["deaths"][index]),
                "efficiency": info["efficiency"][index],
                "flow": info["flow"][index],
            }
            if "tile_names" in static:
                item["tile_names"] = static["tile_names"]
            decoded.append(item)
        return decoded

    def reset(
        self,
        seed=None,
        agents: Optional[list[int]] = None,
        spawns: Optional[list[dict]] = None,
    ):
        """
        Reset the Godot environment.

        Args:
            seed: Unused; kept for the Gymnasium reset signature.
            agents: Optional zero-based agent indices to reset. ``None`` or an
                empty list resets all agents for backward compatibility. The
                response always contains every agent's observation; the reset
                subset is available as ``last_reset_agents``.
            spawns: Optional entries ``{"agent": int, "cell": [x, y],
                "facing": 1|-1}`` selecting clean standable checkpoints. Only
                agents being reset may have a spawn entry; other agents use the
                level's canonical start. Spawns are ignored (with a warning)
                when the game does not advertise support.

        Returns:
            dict: The initial observation data.
        """
        message = {
            "type": "reset",
        }
        if agents is not None:
            if not self.supports_agent_reset:
                print(
                    "warning: game does not advertise per-agent reset; "
                    "resetting all agents",
                    flush=True,
                )
            message["agents"] = [int(index) for index in agents]

        if spawns:
            if not self.supports_spawn_cells:
                print(
                    "warning: game does not advertise spawn-cell reset; "
                    "ignoring spawns",
                    flush=True,
                )
            else:
                message["spawns"] = [
                    _normalise_spawn(entry) for entry in spawns
                ]

        self._send_as_json(message)
        response = self._get_response()
        response["obs"] = self._process_obs(response["obs"])
        assert response["type"] == "reset"
        self._apply_reset_info(response)
        returned_agents = response.get("agents")
        if isinstance(returned_agents, list):
            self.last_reset_agents = [int(index) for index in returned_agents]
        else:
            self.last_reset_agents = list(range(self.num_envs))
        returned_spawns = response.get("spawns")
        if isinstance(returned_spawns, list):
            self.last_reset_spawns = returned_spawns
        elif isinstance(returned_spawns, dict):
            self.last_reset_spawns = [
                {**entry, "agent": int(agent)}
                for agent, entry in returned_spawns.items()
            ]
        else:
            self.last_reset_spawns = []
        obs = response["obs"]
        infos = self._decode_info(response.get("info", []))
        if not isinstance(infos, list):
            infos = [{}] * self.num_envs
        if not infos:
            infos = [{}] * self.num_envs
        return obs, infos

    def call(self, method):
        message = {
            "type": "call",
            "method": method,
        }
        self._send_as_json(message)
        response = self._get_json_dict()

        return response["returns"]

    def close(self):
        message = {
            "type": "close",
        }
        self._send_as_json(message)
        print("close message sent")
        time.sleep(1.0)
        self.connection.close()
        if self.log_handle is not None:
            self.log_handle.flush()
            self.log_handle.close()
            self.log_handle = None
        try:
            atexit.unregister(self._close)
        except Exception as e:
            print("exception unregistering close method", e)

    @property
    def action_space(self):
        """
        Returns a single action space.
        """
        return self.action_space_processor.action_space

    def _close(self):
        print("exit was not clean, using atexit to close env")
        self.close()

    def _launch_env(
        self,
        env_path,
        port,
        show_window,
        framerate,
        seed,
        action_repeat,
        speedup,
        log_path=None,
        env=None,
        **kwargs,
    ):
        # --fixed-fps {framerate}
        path = convert_macos_path(env_path) if platform == "darwin" else env_path

        launch_cmd = f"{path} --port={port} --env_seed={seed}"

        if show_window is False:
            launch_cmd += " --disable-render-loop --headless"
        if framerate is not None:
            launch_cmd += f" --fixed-fps {framerate}"
        if action_repeat is not None:
            launch_cmd += f" --action_repeat={action_repeat}"
        if speedup is not None:
            launch_cmd += f" --speedup={speedup}"
        if len(kwargs) > 0:
            for key, value in kwargs.items():
                launch_cmd += f" --{key}={value}"

        launch_cmd = launch_cmd.split(" ")
        popen_kwargs = {"start_new_session": True}
        if env is not None:
            popen_kwargs["env"] = env
        if log_path:
            self.log_handle = open(log_path, "w", encoding="utf-8")
            popen_kwargs["stdout"] = self.log_handle
            popen_kwargs["stderr"] = subprocess.STDOUT
        self.proc = subprocess.Popen(launch_cmd, **popen_kwargs)

    def _start_server(self):
        # Either launch a an exported Godot project or connect to a playing godot game
        # connect to playing godot game

        print(f"waiting for remote GODOT connection on port {self.port}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Bind the socket to the port, "localhost" was not working on windows VM, had to use the IP
        server_address = ("127.0.0.1", self.port)
        sock.bind(server_address)

        # Listen for incoming connections
        sock.listen(1)
        sock.settimeout(GodotEnv.DEFAULT_TIMEOUT)
        connection, client_address = sock.accept()
        # connection.settimeout(GodotEnv.DEFAULT_TIMEOUT)
        #        connection.setblocking(False) TODO
        print("connection established")
        return connection

    def _handshake(self):
        message = {
            "type": "handshake",
            "major_version": GodotEnv.MAJOR_VERSION,
            "minor_version": GodotEnv.MINOR_VERSION,
        }

        self._send_as_json(message)

    def _get_env_info(self):
        message = {"type": "env_info"}
        self._send_as_json(message)

        json_dict = self._get_json_dict()

        assert json_dict["type"] == "env_info"

        # Number of AIController instances in a single Godot env/process
        self.num_envs = json_dict["n_agents"]
        self.supports_agent_reset = bool(
            json_dict.get("supports_agent_reset", False)
        )
        self.supports_spawn_cells = bool(
            json_dict.get("supports_spawn_cells", False)
        )
        self.supports_state_reset = bool(
            json_dict.get("supports_state_reset", False)
        )
        self.info_schema = int(json_dict.get("info_schema", 1))
        self.static_info = dict(json_dict.get("static_info") or {})
        self._spawn_cells = [[-1, -1] for _ in range(self.num_envs)]

        # actions can be "single" for a single action head
        # or "multi" for several outputeads

        raw_action_space = json_dict["action_space"]
        action_sample = (
            raw_action_space[0]
            if isinstance(raw_action_space, list) and raw_action_space
            else raw_action_space
        )
        print(
            f"action space: {self.num_envs} agents, {action_sample}",
            flush=True,
        )
        # Compatibility with previous versions of Godot plugin:
        # A single action space will be received as a dict in previous versions,
        # A list of dicts will be received from the newer version, defining the action_space for each agent (AIController)
        if isinstance(json_dict["action_space"], dict):
            json_dict["action_space"] = [json_dict["action_space"]] * self.num_envs

        for agent_action_space in json_dict["action_space"]:
            tmp_action_spaces = OrderedDict()
            for k, v in agent_action_space.items():
                if v["action_type"] == "discrete":
                    tmp_action_spaces[k] = spaces.Discrete(v["size"])
                elif v["action_type"] == "continuous":
                    tmp_action_spaces[k] = spaces.Box(low=-1.0, high=1.0, shape=(v["size"],))
                else:
                    print(f"action space {v['action_type']} is not supported")
                    assert 0, f"action space {v['action_type']} is not supported"
            self.action_spaces.append(spaces.Dict(tmp_action_spaces))

        raw_obs_space = json_dict["observation_space"]
        obs_sample = (
            raw_obs_space[0]
            if isinstance(raw_obs_space, list) and raw_obs_space
            else raw_obs_space
        )
        print(
            f"observation space: {self.num_envs} agents, {obs_sample}",
            flush=True,
        )
        # Compatibility with older versions of Godot plugin:
        # A single observation space will be received as a dict in previous versions,
        # A list of dicts will be received from newer version, defining the observation_space for each agent (AIController)
        if isinstance(json_dict["observation_space"], dict):
            json_dict["observation_space"] = [json_dict["observation_space"]] * self.num_envs

        for agent_obs_space in json_dict["observation_space"]:
            observation_spaces = {}
            for k, v in agent_obs_space.items():
                if v["space"] == "box":
                    if "2d" in k:
                        observation_spaces[k] = spaces.Box(
                            low=0,
                            high=255,
                            shape=v["size"],
                            dtype=np.uint8,
                        )
                    else:
                        observation_spaces[k] = spaces.Box(
                            low=-1.0,
                            high=1.0,
                            shape=v["size"],
                            dtype=np.float32,
                        )
                elif v["space"] == "discrete":
                    observation_spaces[k] = spaces.Discrete(v["size"])
                else:
                    print(f"observation space {v['space']} is not supported")
                    assert 0, f"observation space {v['space']} is not supported"
            self.observation_spaces.append(spaces.Dict(observation_spaces))

        # Gets policy names defined in AIControllers in Godot. If an older version of the plugin is used and no policy
        # names are sent, "shared_policy" will be set for compatibility.
        self.agent_policy_names = json_dict.get("agent_policy_names", ["shared_policy"] * self.num_envs)

    @staticmethod
    def _decode_2d_obs_from_string(
        hex_string,
        shape,
    ):
        return np.frombuffer(bytes.fromhex(hex_string), dtype=np.uint8).reshape(shape)

    def _send_as_json(self, dictionary):
        message_json = json.dumps(dictionary)
        self._send_string(message_json)

    def _get_json_dict(self):
        return self._get_response()

    def _get_response(self):
        data = self._get_data()
        if data is None:
            raise RuntimeError("Godot connection closed while reading a response")
        if data[:1] == b"B":
            response = self._decode_binary_response(data)
        else:
            response = json.loads(data.decode())
        self.last_profile = response.get("profile", {})
        return response

    def _decode_binary_response(self, data: bytes) -> dict:
        if len(data) < 5:
            raise ValueError("truncated binary response")
        header_size = int.from_bytes(data[1:5], "little")
        header_start = 5
        raw_start = header_start + header_size
        header = json.loads(data[header_start:raw_start].decode())
        count = int(header["obs_count"])
        size = int(header["obs_size"])
        values = np.frombuffer(data[raw_start:], dtype="<f4")
        if values.size != count * size:
            raise ValueError(
                f"binary observation size mismatch: {values.size} != {count * size}"
            )
        values = values.reshape(count, size)
        header["obs"] = [{"obs": values[index]} for index in range(count)]
        return header

    def _get_obs(self):
        return self._get_data()

    def _clear_socket(self):
        self.connection.setblocking(False)
        try:
            while True:
                data = self.connection.recv(4)
                if not data:
                    break
        except BlockingIOError:
            pass
        self.connection.setblocking(True)

    def _get_data(self):
        try:
            # Receive the size (in bytes) of the remaining data to receive
            string_size_bytes: bytearray = bytearray()
            received_length: int = 0

            # The first 4 bytes contain the length of the remaining data
            length: int = 4

            while received_length < length:
                data = self.connection.recv(length - received_length)
                received_length += len(data)
                string_size_bytes.extend(data)

            length = int.from_bytes(string_size_bytes, "little")

            # Receive the rest of the data
            payload: bytearray = bytearray()
            received_length = 0

            while received_length < length:
                data = self.connection.recv(length - received_length)
                received_length += len(data)
                payload.extend(data)

            return bytes(payload)
        except socket.timeout as e:
            print("env timed out", e)
        return None

    def _send_string(self, string):
        encoded = string.encode()
        message = len(encoded).to_bytes(4, "little") + encoded
        self.connection.sendall(message)

    def _send_action(self, action):
        self._send_string(action)


def interactive():
    env = GodotEnv()
    print("observation space", env.observation_space)
    print("action space", env.action_space)

    obs = env.reset()
    for i in range(1000):
        action = [env.action_space.sample() for _ in range(env.num_envs)]
        action = list(zip(*action))

        obs, reward, term, trunc, info = env.step(action)
    env.close()


if __name__ == "__main__":
    interactive()
