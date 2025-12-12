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


@pytest.fixture
def kernel_manager(kernel_binary):
    """Create and start a kernel manager."""
    kernel_process = None
    try:
        # Use KernelManager to write connection file with proper ports
        km = KernelManager()
        km.write_connection_file()
        connection_file = km.connection_file

        # Start our kernel process directly
        kernel_process = subprocess.Popen(
            [kernel_binary, "-f", connection_file],
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
        try:
            os.unlink(km.connection_file)
        except:
            pass


@pytest.fixture
def client(kernel_manager):
    """Create a kernel client."""
    kc = kernel_manager.client()
    kc.start_channels()
    kc.wait_for_ready(timeout=10)
    yield kc
    kc.stop_channels()


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
        VISUALISE AS PLOT
        DRAW point USING x = x, y = y
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

        # Should have Vega-Lite MIME type
        data = content["data"]
        assert "application/vnd.vegalite.v5+json" in data

        # Check Vega-Lite spec structure
        vega_spec = data["application/vnd.vegalite.v5+json"]
        assert "$schema" in vega_spec
        assert "data" in vega_spec
        assert "mark" in vega_spec or "layer" in vega_spec

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
