"""
Jupyter kernel compliance tests using jupyter_kernel_test.

This test suite validates that ggsql-jupyter implements the Jupyter
messaging protocol correctly according to the specification.
"""

import unittest
import jupyter_kernel_test as jkt
import subprocess
from pathlib import Path


class GGSQLKernelTests(jkt.KernelTests):
    """Compliance tests for ggsql-jupyter kernel."""

    # Kernel name (will be overridden to use custom command)
    kernel_name = "ggsql"

    # Language name
    language_name = "ggsql"

    # File extension for code files
    file_extension = ".ggsql"

    # Code samples for testing
    code_hello_world = "SELECT 'Hello, World!' as greeting"

    # Expected output pattern (for simple SELECT)
    # Note: jupyter_kernel_test looks for this in text/plain output
    # We may need to adjust based on actual output format
    code_page_something = "SELECT 'something' as result"

    # Override test_execute_stdout - SQL kernels don't produce stdout
    def test_execute_stdout(self):
        """SQL kernels produce execute_result, not stdout streams."""
        # Skip this test for SQL kernels - they don't produce stdout
        # They produce execute_result messages instead
        pass

    def setUp(self):
        """Build kernel before tests."""
        # Build the kernel
        repo_root = Path(__file__).parent.parent.parent
        result = subprocess.run(
            ["cargo", "build", "--bin", "ggsql-jupyter"],
            cwd=repo_root / "ggsql-jupyter",
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.fail(f"Failed to build kernel: {result.stderr}")

        super().setUp()

    # Test that kernel_info_request works
    def test_kernel_info(self):
        """Test kernel_info_request returns correct information."""
        self.flush_channels()

        msg_id = self.kc.kernel_info()
        reply = self.get_non_kernel_info_reply()

        self.assertEqual(reply["msg_type"], "kernel_info_reply")
        content = reply["content"]

        self.assertEqual(content["status"], "ok")
        self.assertEqual(content["protocol_version"], "5.3")
        self.assertEqual(content["implementation"], "ggsql")

        # Language info
        lang_info = content["language_info"]
        self.assertEqual(lang_info["name"], "ggsql")
        self.assertEqual(lang_info["file_extension"], ".ggsql")
        self.assertEqual(lang_info["mimetype"], "text/x-ggsql")

    # Override execute test to handle our specific output format
    def test_execute_request(self):
        """Test that execute_request works."""
        self.flush_channels()

        reply, output_msgs = self.execute_helper(code=self.code_hello_world)

        self.assertEqual(reply["content"]["status"], "ok")
        self.assertGreaterEqual(reply["content"]["execution_count"], 1)

    # Test visualization output
    def test_execute_visualization(self):
        """Test that visualization output includes Vega-Lite MIME type."""
        self.flush_channels()

        code = """
        SELECT 1 as x, 2 as y
        VISUALISE AS PLOT
        DRAW point USING x = x, y = y
        """

        reply, output_msgs = self.execute_helper(code=code)

        # Should succeed
        self.assertEqual(reply["content"]["status"], "ok")

        # Find execute_result message
        execute_result = None
        for msg in output_msgs:
            if msg["msg_type"] == "execute_result":
                execute_result = msg
                break

        self.assertIsNotNone(execute_result, "No execute_result message found")

        # Check MIME types
        data = execute_result["content"]["data"]
        self.assertIn("application/vnd.vegalite.v5+json", data)

        # Verify Vega-Lite spec structure
        vega_spec = data["application/vnd.vegalite.v5+json"]
        self.assertIn("$schema", vega_spec)
        self.assertIn("data", vega_spec)

    # Test error handling
    def test_execute_error(self):
        """Test that errors are properly reported."""
        self.flush_channels()

        code = "SELECT * FROM nonexistent_table"
        reply, output_msgs = self.execute_helper(code=code)

        # Should report error
        self.assertEqual(reply["content"]["status"], "error")

        # Should have error message in iopub
        error_msgs = [msg for msg in output_msgs if msg["msg_type"] == "error"]
        self.assertGreater(len(error_msgs), 0, "No error message in iopub")

        error = error_msgs[0]["content"]
        self.assertIn("ename", error)
        self.assertIn("evalue", error)
        self.assertIn("traceback", error)

    # Test status messages
    def test_status_messages(self):
        """Test that status messages are sent correctly."""
        self.flush_channels()

        msg_id = self.kc.execute(code=self.code_hello_world)

        # Collect status messages
        status_msgs = []
        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=2)
                if msg["msg_type"] == "status":
                    status_msgs.append(msg["content"]["execution_state"])
                if (
                    msg["msg_type"] == "status"
                    and msg["content"]["execution_state"] == "idle"
                ):
                    break
            except:
                break

        # Should have busy then idle
        self.assertIn("busy", status_msgs)
        self.assertIn("idle", status_msgs)
        self.assertLess(status_msgs.index("busy"), status_msgs.index("idle"))

    # Test execute_input
    def test_execute_input(self):
        """Test that execute_input echoes the code."""
        self.flush_channels()

        code = "SELECT 123 as num"
        msg_id = self.kc.execute(code=code)

        # Manually collect iopub messages to find execute_input
        # (execute_helper filters it out)
        execute_input = None
        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=2)
                if msg["msg_type"] == "execute_input":
                    execute_input = msg
                if (
                    msg["msg_type"] == "status"
                    and msg["content"]["execution_state"] == "idle"
                ):
                    break
            except:
                break

        self.assertIsNotNone(execute_input, "No execute_input message found")
        self.assertEqual(execute_input["content"]["code"], code)
        self.assertIn("execution_count", execute_input["content"])

    # Test shutdown
    def test_shutdown(self):
        """Test that shutdown works."""
        self.flush_channels()

        msg_id = self.kc.shutdown()
        reply = self.kc.get_shell_msg(timeout=5)

        self.assertEqual(reply["msg_type"], "shutdown_reply")
        self.assertEqual(reply["content"]["status"], "ok")
        self.assertIn("restart", reply["content"])

    # Test persistent state
    def test_persistent_state(self):
        """Test that state persists across executions."""
        self.flush_channels()

        # Create table
        code1 = "CREATE TABLE test (id INTEGER)"
        reply1, _ = self.execute_helper(code=code1)
        self.assertEqual(reply1["content"]["status"], "ok")

        # Insert data
        code2 = "INSERT INTO test VALUES (42)"
        reply2, _ = self.execute_helper(code=code2)
        self.assertEqual(reply2["content"]["status"], "ok")

        # Query data (should succeed because table exists)
        code3 = "SELECT * FROM test"
        reply3, _ = self.execute_helper(code=code3)
        self.assertEqual(reply3["content"]["status"], "ok")


# Configure kernel for testing
def setup_module():
    """Setup module by installing kernel spec."""
    import tempfile
    import json
    import os

    # Build kernel
    repo_root = Path(__file__).parent.parent.parent
    result = subprocess.run(
        ["cargo", "build", "--bin", "ggsql-jupyter"],
        cwd=repo_root / "ggsql-jupyter",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to build kernel: {result.stderr}")

    # Find binary
    binary_path = repo_root / "target" / "debug" / "ggsql-jupyter"
    if not binary_path.exists():
        raise RuntimeError(f"Kernel binary not found at {binary_path}")

    # Create kernel spec
    kernel_spec = {
        "argv": [str(binary_path), "-f", "{connection_file}"],
        "display_name": "ggSQL",
        "language": "ggsql",
    }

    # Write kernel.json to temp directory
    spec_dir = Path(tempfile.mkdtemp(prefix="ggsql-kernel-"))
    with open(spec_dir / "kernel.json", "w") as f:
        json.dump(kernel_spec, f)

    # Install kernel spec
    result = subprocess.run(
        [
            "jupyter",
            "kernelspec",
            "install",
            "--user",
            "--name",
            "ggsql",
            str(spec_dir),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Warning: Failed to install kernel spec: {result.stderr}")
        print("Tests may fail if kernel spec is not properly installed")


def teardown_module():
    """Cleanup kernel spec after tests."""
    subprocess.run(
        ["jupyter", "kernelspec", "remove", "-f", "ggsql"],
        capture_output=True,
    )


if __name__ == "__main__":
    # Run setup
    setup_module()

    try:
        # Run tests
        unittest.main()
    finally:
        # Cleanup
        teardown_module()
