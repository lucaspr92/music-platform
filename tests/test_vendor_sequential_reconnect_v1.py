from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
import types


def test_vendor_bridge_serves_sequential_clients_without_sleep(monkeypatch) -> None:
    """The vendored one-client-at-a-time contract must survive A -> B -> C -> D."""
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    monkeypatch.setenv("ABLETON_MCP_PORT", str(port))

    framework = types.ModuleType("_Framework")
    control_surface_module = types.ModuleType("_Framework.ControlSurface")

    class ControlSurface:
        def __init__(self, c_instance=None):
            del c_instance

        def song(self):
            return object()

        def log_message(self, message):
            del message

        def show_message(self, message):
            del message

        def disconnect(self):
            return None

    control_surface_module.ControlSurface = ControlSurface
    monkeypatch.setitem(sys.modules, "_Framework", framework)
    monkeypatch.setitem(sys.modules, "_Framework.ControlSurface", control_surface_module)

    spec = importlib.util.spec_from_file_location(
        "vendor_bridge_sequential_test",
        "vendor/abletonmcp_remote_script/AbletonMCP/__init__.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Harness(module.AbletonMCP):
        def _process_command(self, command):
            return {
                "status": "success",
                "request_id": command.get("request_id"),
                "result": {"protocol_version": "1"},
            }

    bridge = Harness.__new__(Harness)
    bridge.server = None
    bridge.client_threads = []
    bridge._threads_lock = threading.Lock()
    bridge.server_thread = None
    bridge.running = False
    # __init__ is bypassed, so supply the transport state the client handler
    # touches in its finally block; without it the handler thread dies on
    # cleanup and the one-client-at-a-time contract is never really exercised.
    bridge._transport_event_lock = threading.RLock()
    bridge._transport_subscribers = {}
    bridge._transport_event_capable = False
    bridge.start_server()

    def empty_connectivity_probe() -> None:
        client = socket.socket()
        client.settimeout(2.0)
        client.connect(("127.0.0.1", port))
        client.close()

    try:
        responses = []
        for label in "ABCD":
            # This mirrors tcp_connectable() immediately before a real probe.
            empty_connectivity_probe()
            client = socket.socket()
            client.settimeout(2.0)
            client.connect(("127.0.0.1", port))
            client.sendall(
                json.dumps(
                    {
                        "type": "protocol_hello",
                        "params": {},
                        "request_id": f"req_{label}",
                    }
                ).encode("utf-8")
            )
            responses.append(json.loads(client.recv(8192).decode("utf-8")))
            client.close()

        assert [row["request_id"] for row in responses] == [
            "req_A",
            "req_B",
            "req_C",
            "req_D",
        ]
        assert all(row["status"] == "success" for row in responses)
    finally:
        bridge.running = False
        if bridge.server is not None:
            bridge.server.close()
        if bridge.server_thread is not None:
            bridge.server_thread.join(2.0)
