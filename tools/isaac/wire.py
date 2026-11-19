"""Tiny length-prefixed wire format between the Isaac Lab world (Python 3.12) and the FORESIGHT brain (habitat env,
Python 3.9): a JSON header plus an optional raw byte payload. No pickle, so the two interpreters never need
compatible library versions."""
import json
import socket
import struct

PORT = 47011


def send(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    h = json.dumps(header).encode()
    sock.sendall(struct.pack("!II", len(h), len(payload)) + h + payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def recv(sock: socket.socket):
    nh, np_ = struct.unpack("!II", _recv_exact(sock, 8))
    header = json.loads(_recv_exact(sock, nh).decode())
    return header, _recv_exact(sock, np_) if np_ else b""
