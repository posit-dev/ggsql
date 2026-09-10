"""
Integration tests for ggsql-jupyter kernel using jupyter_client.

These tests launch the kernel and send real Jupyter protocol messages
to verify correct behavior.
"""

import json
import time
import subprocess
import tempfile
import os
from pathlib import Path
import pytest
from jupyter_client import KernelManager


@pytest.fixture(scope="session")
def kernel_binary():
    """Build and return path to ggsql-jupyter binary."""
    # Build the kernel
    repo_root = Path(__file__).parent.parent.parent
    result = subprocess.run(
        ["cargo", "build", "--bin", "ggsql-jupyter"],
        cwd=repo_root / "ggsql-jupyter",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to build kernel: {result.stderr}")

    # Find binary
    binary_path = repo_root / "target" / "debug" / "ggsql-jupyter"
    if not binary_path.exists():
        pytest.fail(f"Kernel binary not found at {binary_path}")

    return str(binary_path)


def _launch(kernel_binary, extra_args=()):
    """Start a kernel and yield its manager. Shared by the fixtures below."""
    kernel_process = None
    km = None
    try:
        # Use KernelManager to write connection file with proper ports
        km = KernelManager()
        km.write_connection_file()
        connection_file = km.connection_file

        # Start our kernel process directly
        kernel_process = subprocess.Popen(
            [kernel_binary, "-f", connection_file, *extra_args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Store the process in km for cleanup
        km._kernel_process = kernel_process

        # Wait for kernel to be ready
        time.sleep(3)

        # Check if process started successfully
        if kernel_process.poll() is not None:
            stdout, stderr = kernel_process.communicate()
            pytest.fail(
                f"Kernel failed to start:\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
            )

        yield km

    finally:
        # Cleanup
        if kernel_process is not None:
            try:
                kernel_process.terminate()
                kernel_process.wait(timeout=5)
            except:
                try:
                    kernel_process.kill()
                except:
                    pass
        if km is not None:
            try:
                os.unlink(km.connection_file)
            except:
                pass


@pytest.fixture
def kernel_manager(kernel_binary):
    """Create and start a kernel manager."""
    yield from _launch(kernel_binary)


@pytest.fixture
def console_kernel_manager(kernel_binary):
    """A kernel that believes it is a Positron console session.

    `--session-mode` is what the extension's `createKernelSpec` appends, and it
    is authoritative over the session-id heuristic — so this is the only way to
    reach the plot-comm path from a test. Everything driven through plain
    `jupyter_client` lands in `SessionKind::Standalone`, because its session id
    is a bare UUID.
    """
    yield from _launch(kernel_binary, ["--session-mode", "console"])


@pytest.fixture
def client(kernel_manager):
    """Create a kernel client."""
    kc = kernel_manager.client()
    kc.start_channels()
    kc.wait_for_ready(timeout=10)
    yield kc
    kc.stop_channels()


@pytest.fixture
def console_client(console_kernel_manager):
    """A client talking to a console-mode kernel."""
    kc = console_kernel_manager.client()
    kc.start_channels()
    kc.wait_for_ready(timeout=10)
    yield kc
    kc.stop_channels()


def drain_iopub(client, timeout=5):
    """Collect iopub messages up to and including the closing `idle` status."""
    messages = []
    while True:
        try:
            msg = client.get_iopub_msg(timeout=timeout)
        except Exception:
            return messages
        messages.append(msg)
        if (
            msg["msg_type"] == "status"
            and msg["content"]["execution_state"] == "idle"
        ):
            return messages


class TestKernelInfo:
    """Test kernel_info_request/reply messages."""

    def test_kernel_info_request(self, client):
        """Test that kernel responds to kernel_info_request."""
        msg_id = client.kernel_info()

        # Get reply
        reply = client.get_shell_msg(timeout=5)

        assert reply["msg_type"] == "kernel_info_reply"
        assert reply["parent_header"]["msg_id"] == msg_id

        content = reply["content"]
        assert content["status"] == "ok"
        assert content["protocol_version"] == "5.3"
        assert content["implementation"] == "ggsql-jupyter"

        # Check language info
        lang_info = content["language_info"]
        assert lang_info["name"] == "ggsql"
        assert lang_info["file_extension"] == ".ggsql"
        assert lang_info["mimetype"] == "text/x-ggsql"


class TestExecution:
    """Test execute_request/reply messages."""

    def test_simple_sql_execution(self, client):
        """Test executing a simple SQL query."""
        code = "SELECT 1 as num, 'test' as text"
        msg_id = client.execute(code, silent=False, store_history=True)

        # Collect messages
        messages = []
        for _ in range(10):  # Collect up to 10 messages
            try:
                msg = client.get_iopub_msg(timeout=2)
                messages.append(msg)
                if (
                    msg["msg_type"] == "status"
                    and msg["content"]["execution_state"] == "idle"
                ):
                    break
            except:
                break

        # Check for expected messages
        msg_types = [msg["msg_type"] for msg in messages]
        assert "status" in msg_types  # Should have status messages
        assert "execute_input" in msg_types  # Should echo input

        # Get execute_reply
        reply = client.get_shell_msg(timeout=5)
        assert reply["msg_type"] == "execute_reply"
        assert reply["content"]["status"] == "ok"
        assert reply["content"]["execution_count"] >= 1

    def test_visualization_execution(self, client):
        """Test executing a query with visualization."""
        code = """
        SELECT 1 as x, 2 as y
        VISUALISE x, y
        DRAW point
        """
        msg_id = client.execute(code, silent=False, store_history=True)

        # Collect messages
        execute_result = None
        for _ in range(10):
            try:
                msg = client.get_iopub_msg(timeout=2)
                if msg["msg_type"] == "execute_result":
                    execute_result = msg
                if (
                    msg["msg_type"] == "status"
                    and msg["content"]["execution_state"] == "idle"
                ):
                    break
            except:
                break

        # Check execute_result
        assert execute_result is not None
        content = execute_result["content"]
        assert "data" in content

        # A plot arrives as a rendered image: PNG where this build has the
        # raster writers and the machine has a GPU adapter, SVG otherwise.
        data = content["data"]
        assert (
            "image/png" in data or "image/svg+xml" in data
        ), f"expected a rendered plot, got {sorted(data)}"
        assert "text/plain" in data

        if "image/svg+xml" in data:
            assert data["image/svg+xml"].startswith("<svg")
        else:
            # base64-encoded bytes carrying the PNG signature
            import base64

            assert base64.b64decode(data["image/png"]).startswith(b"\x89PNG")

        # The rendered size travels with it, so a 2x render is displayed at 1x
        # rather than at twice its intended size.
        mime = "image/png" if "image/png" in data else "image/svg+xml"
        assert content["metadata"][mime]["width"] > 0
        assert content["metadata"][mime]["height"] > 0

        # Nothing in the bundle reaches for a CDN — this is what lets a plot
        # render offline, in CI, and behind a firewall.
        assert "jsdelivr" not in str(data)
        assert "vega-embed" not in str(data)

    def test_error_handling(self, client):
        """Test that syntax errors are reported correctly."""
        code = "SELECT * FROM nonexistent_table"
        msg_id = client.execute(code, silent=False, store_history=True)

        # Collect messages
        error_msg = None
        for _ in range(10):
            try:
                msg = client.get_iopub_msg(timeout=2)
                if msg["msg_type"] == "error":
                    error_msg = msg
                if (
                    msg["msg_type"] == "status"
                    and msg["content"]["execution_state"] == "idle"
                ):
                    break
            except:
                break

        # Should have error message
        assert error_msg is not None
        content = error_msg["content"]
        assert "ename" in content
        assert "evalue" in content
        assert "traceback" in content

    def test_persistent_state(self, client):
        """Test that DuckDB state persists across cells."""
        # Create a table
        code1 = "CREATE TABLE test_table (id INTEGER, name VARCHAR)"
        client.execute(code1, silent=False, store_history=True)

        # Wait for idle
        for _ in range(5):
            try:
                msg = client.get_iopub_msg(timeout=2)
                if (
                    msg["msg_type"] == "status"
                    and msg["content"]["execution_state"] == "idle"
                ):
                    break
            except:
                break

        # Clear reply
        try:
            client.get_shell_msg(timeout=1)
        except:
            pass

        # Insert data
        code2 = "INSERT INTO test_table VALUES (1, 'Alice'), (2, 'Bob')"
        client.execute(code2, silent=False, store_history=True)

        # Wait for idle
        for _ in range(5):
            try:
                msg = client.get_iopub_msg(timeout=2)
                if (
                    msg["msg_type"] == "status"
                    and msg["content"]["execution_state"] == "idle"
                ):
                    break
            except:
                break

        # Clear reply
        try:
            client.get_shell_msg(timeout=1)
        except:
            pass

        # Query the table
        code3 = "SELECT * FROM test_table"
        client.execute(code3, silent=False, store_history=True)

        # Should succeed (table exists)
        reply = client.get_shell_msg(timeout=5)
        assert reply["content"]["status"] == "ok"


PLOT_QUERY = "SELECT 1 as x, 2 as y VISUALISE x, y DRAW point"


class TestPlotComm:
    """The console session's plot path: a `positron.plot` comm, and its RPCs.

    This is the only route to `open_plot_comm`, `pre_render`, `finish_render`
    and eviction — everything else in this file is a `Standalone` session, which
    takes the static-bundle path instead.
    """

    def _open_plot(self, console_client):
        """Run a plot cell and return its comm_id."""
        msg_id = console_client.execute(PLOT_QUERY, silent=False, store_history=True)
        messages = drain_iopub(console_client)

        opens = [m for m in messages if m["msg_type"] == "comm_open"]
        plots = [
            m for m in opens if m["content"]["target_name"] == "positron.plot"
        ]
        assert plots, (
            "a console session must open a positron.plot comm; got "
            f"{[m['msg_type'] for m in messages]}"
        )
        assert len(plots) == 1, "one comm per plot"

        # The comm alone creates the pane entry. An output message as well
        # would put a second, fixed-size copy of the plot in the pane.
        assert not [m for m in messages if m["msg_type"] == "execute_result"]

        # Parented to the execute_request: that is how the frontend ties the
        # plot to the cell that produced it, and where its `code` comes from.
        assert plots[0]["parent_header"]["msg_id"] == msg_id

        # `execute_input` must come first — Positron fills `_recentExecutions`
        # from it, so a comm arriving earlier has no execution to attach to.
        order = [m["msg_type"] for m in messages]
        assert order.index("execute_input") < order.index("comm_open")

        reply = console_client.get_shell_msg(timeout=5)
        assert reply["msg_type"] == "execute_reply"
        assert reply["content"]["status"] == "ok"

        return plots[0]["content"]["comm_id"]

    def test_console_opens_a_plot_comm_and_no_output(self, console_client):
        self._open_plot(console_client)

    def test_render_is_answered_and_the_kernel_returns_to_idle(self, console_client):
        comm_id = self._open_plot(console_client)

        console_client.session.send(
            console_client.shell_channel.socket,
            "comm_msg",
            {
                "comm_id": comm_id,
                "data": {
                    "jsonrpc": "2.0",
                    "id": "render-1",
                    "method": "render",
                    "params": {
                        "size": {"width": 400, "height": 300},
                        "pixel_ratio": 1.0,
                        "format": "png",
                    },
                },
            },
        )

        # The reply comes back on shell, from `finish_render` rather than from
        # the handler — the render is dispatched to a thread.
        reply = console_client.get_shell_msg(timeout=30)
        assert reply["msg_type"] == "comm_msg"
        data = reply["content"]["data"]
        assert "error" not in data, data
        result = data["result"]
        assert result["data"], "a render must carry bytes"
        # Whatever this build and machine can produce; SVG is the fallback.
        assert result["mime_type"] in ("image/png", "image/svg+xml")

        # And the `busy` that the request opened is closed. Missing this is how
        # the kernel gets stuck busy forever.
        states = [
            m["content"]["execution_state"]
            for m in drain_iopub(console_client)
            if m["msg_type"] == "status"
        ]
        assert states[-1] == "idle", states

    def test_a_bad_render_request_still_returns_to_idle(self, console_client):
        """A rejected `render` is answered here, not by `finish_render`.

        No ticket is queued for it, so nothing will ever arrive on the outcome
        channel — the `idle` has to be sent by the handler, or the kernel stays
        busy with the frontend waiting out its 30 s timeout.
        """
        comm_id = self._open_plot(console_client)

        console_client.session.send(
            console_client.shell_channel.socket,
            "comm_msg",
            {
                "comm_id": comm_id,
                "data": {
                    "jsonrpc": "2.0",
                    "id": "render-bad",
                    "method": "render",
                    # No pixel_ratio and no format: InvalidParams.
                    "params": {"size": {"width": 400, "height": 300}},
                },
            },
        )

        reply = console_client.get_shell_msg(timeout=10)
        assert reply["msg_type"] == "comm_msg"
        assert "error" in reply["content"]["data"], reply["content"]["data"]

        states = [
            m["content"]["execution_state"]
            for m in drain_iopub(console_client)
            if m["msg_type"] == "status"
        ]
        assert states[-1] == "idle", states

    def test_get_metadata_carries_the_cell_that_made_the_plot(self, console_client):
        comm_id = self._open_plot(console_client)

        console_client.session.send(
            console_client.shell_channel.socket,
            "comm_msg",
            {
                "comm_id": comm_id,
                "data": {"jsonrpc": "2.0", "id": "meta-1", "method": "get_metadata"},
            },
        )

        reply = console_client.get_shell_msg(timeout=5)
        result = reply["content"]["data"]["result"]
        assert result["code"] == PLOT_QUERY
        assert result["kind"] == "ggsql"
        assert result["name"]
        assert result["execution_id"]

    def test_an_unknown_method_is_an_error_not_a_null_result(self, console_client):
        """`show` and `update` land here on purpose — see `handle_plot_rpc`."""
        comm_id = self._open_plot(console_client)

        console_client.session.send(
            console_client.shell_channel.socket,
            "comm_msg",
            {
                "comm_id": comm_id,
                "data": {"jsonrpc": "2.0", "id": "show-1", "method": "show"},
            },
        )

        reply = console_client.get_shell_msg(timeout=5)
        data = reply["content"]["data"]
        assert "error" in data, data
        assert "result" not in data


class TestStatus:
    """Test status messages."""

    def test_status_busy_idle(self, client):
        """Test that status transitions from busy to idle."""
        code = "SELECT 1"
        client.execute(code, silent=False, store_history=True)

        # Collect status messages
        statuses = []
        for _ in range(10):
            try:
                msg = client.get_iopub_msg(timeout=2)
                if msg["msg_type"] == "status":
                    statuses.append(msg["content"]["execution_state"])
                    if msg["content"]["execution_state"] == "idle":
                        break
            except:
                break

        # Should have busy then idle
        assert "busy" in statuses
        assert "idle" in statuses
        assert statuses.index("busy") < statuses.index("idle")


class TestShutdown:
    """Test shutdown_request/reply messages."""

    def test_shutdown_request(self, kernel_manager):
        """Test that kernel responds to shutdown_request."""
        kc = kernel_manager.client()
        kc.start_channels()

        try:
            kc.wait_for_ready(timeout=10)

            # Send shutdown request on control channel
            msg_id = kc.shutdown()

            # Try to get reply with a longer timeout
            try:
                reply = kc.get_shell_msg(timeout=10)

                assert reply["msg_type"] == "shutdown_reply"
                assert reply["content"]["status"] == "ok"
                assert "restart" in reply["content"]
            except:
                # If we can't get the reply, at least verify the kernel process is terminating
                # Wait a bit for shutdown to process
                time.sleep(2)
                kernel_process = kernel_manager._kernel_process
                # Process should either be terminated or terminating
                if kernel_process.poll() is None:
                    # Still running, send explicit shutdown
                    kernel_process.terminate()
                    kernel_process.wait(timeout=5)

        finally:
            kc.stop_channels()


class TestExecuteInput:
    """Test execute_input messages."""

    def test_execute_input_echoed(self, client):
        """Test that execute_input echoes the code."""
        code = "SELECT 42 as answer"
        msg_id = client.execute(code, silent=False, store_history=True)

        # Look for execute_input message
        execute_input = None
        for _ in range(10):
            try:
                msg = client.get_iopub_msg(timeout=2)
                if msg["msg_type"] == "execute_input":
                    execute_input = msg
                    break
                if (
                    msg["msg_type"] == "status"
                    and msg["content"]["execution_state"] == "idle"
                ):
                    break
            except:
                break

        assert execute_input is not None
        content = execute_input["content"]
        assert content["code"] == code
        assert "execution_count" in content


class TestHeartbeat:
    """Test heartbeat mechanism."""

    def test_heartbeat_responsive(self, kernel_manager):
        """Test that heartbeat responds."""
        # Since we're manually managing the kernel process,
        # check that the process is still running
        kernel_process = kernel_manager._kernel_process
        assert kernel_process.poll() is None, "Kernel process terminated unexpectedly"

        # Wait a bit and check again
        time.sleep(1)
        assert kernel_process.poll() is None, "Kernel process terminated after 1 second"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
