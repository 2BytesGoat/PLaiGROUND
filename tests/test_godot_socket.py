"""Tests for the low-level Godot socket framing and failure modes."""

from __future__ import annotations

import socket
import unittest

from plaiground.envs.godot import GodotEnv


class FakeSocket:
    def __init__(self, chunks: list[bytes], error: str | None = None):
        self.chunks = list(chunks)
        self.error = error
        self.received: list[int] = []

    def recv(self, count: int) -> bytes:
        self.received.append(count)
        if self.error == "closed":
            return b""
        if self.error == "timeout":
            raise socket.timeout("fake timeout")
        if not self.chunks:
            return b""
        chunk = self.chunks.pop(0)
        if len(chunk) > count:
            self.chunks.insert(0, chunk[count:])
            chunk = chunk[:count]
        return chunk


def _env(sock: FakeSocket) -> GodotEnv:
    env = object.__new__(GodotEnv)
    env.connection = sock
    env._timeout = 1.0
    return env


class GodotSocketTest(unittest.TestCase):
    def test_get_data_reads_length_prefixed_payload(self) -> None:
        sock = FakeSocket([b"\x03\x00\x00\x00", b"abc"])
        self.assertEqual(_env(sock)._get_data(), b"abc")
        self.assertEqual(sock.received, [4, 3])

    def test_closed_socket_raises_instead_of_looping(self) -> None:
        env = _env(FakeSocket([], error="closed"))
        with self.assertRaises(ConnectionError):
            env._get_data()

    def test_timeout_is_reported(self) -> None:
        env = _env(FakeSocket([], error="timeout"))
        with self.assertRaises(TimeoutError):
            env._get_data()

    def test_partial_frame_then_close_is_reported(self) -> None:
        env = _env(FakeSocket([b"\x03\x00\x00\x00", b"a"]))
        with self.assertRaises(ConnectionError):
            env._get_data()

    def test_invalid_length_is_rejected(self) -> None:
        env = _env(FakeSocket([b"\xff\xff\xff\xff"]))
        with self.assertRaises(RuntimeError):
            env._get_data()


if __name__ == "__main__":
    unittest.main()
