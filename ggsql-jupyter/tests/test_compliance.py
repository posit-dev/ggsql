"""
Jupyter kernel compliance tests using jupyter_kernel_test.

This test suite validates that ggsql-jupyter implements the Jupyter
messaging protocol correctly according to the specification.
"""

import os
import unittest
import jupyter_kernel_test as jkt
import subprocess
from pathlib import Path

from conftest import build_kernel_binary

# Isolated from any real Jupyter install: setup_module points JUPYTER_DATA_DIR
# at a scratch directory before this name is ever installed or removed.
KERNEL_NAME = "ggsql-test"


class ggsqlKernelTests(jkt.KernelTests):
    """Compliance tests for ggsql-jupyter kernel."""

    # Kernel name (will be overridden to use custom command)
    kernel_name = KERNEL_NAME

    # Language name
    language_name = "ggsql"

    # File extension for code files
    file_extension = ".ggsql"

    # Code samples for testing
    code_hello_world = "SELECT 'Hello, World!' as greeting"

    # These have a real implementation behind them, so defining the sample
    # lets jupyter_kernel_test's own inherited test exercise it directly
    # rather than duplicating the assertions in a test we wrote ourselves.
    code_generate_error = "SELECT * FROM nonexistent_table"
    code_execute_result = [{"code": "SELECT 123 as num", "mime": "text/plain"}]
    complete_code_samples = ["SELECT 1"]
    incomplete_code_samples = ["SELECT (1"]

    # Override test_execute_stdout - SQL kernels don't produce stdout
    def test_execute_stdout(self):
        """SQL kernels produce execute_result, not stdout streams."""
        # Skip this test for SQL kernels - they don't produce stdout
        # They produce execute_result messages instead
        pass

    # Everything below has no backing implementation in the kernel: there is
    # no complete_request, inspect_request or history_request handler (see
    # kernel.rs's message dispatch), `payload` is hardcoded to `[]` so there
    # is no pager support, and nothing is ever emitted as `display_data` —
    # results always go out as `execute_result`. Defining the sample
    # attributes that would make these inherited tests run would exercise
    # protocol features this kernel doesn't have, so they're overridden here
    # to record that as a deliberate choice rather than a silent SkipTest.
    def test_execute_stderr(self):
        """No stream messages of any kind are ever emitted."""
        pass

    def test_completion(self):
        """No complete_request handler exists."""
        pass

    def test_pager(self):
        """`payload` is hardcoded to `[]`; there is no pager support."""
        pass

    def test_display_data(self):
        """Results always go out as execute_result, never display_data."""
        pass

    def test_history(self):
        """No history_request handler exists."""
        pass

    def test_inspect(self):
        """No inspect_request handler exists."""
        pass

    # Test that kernel_info_request works
    def test_kernel_info(self):
        """Test kernel_info_request returns correct information.

        `get_non_kernel_info_reply` (jkt's own `execute_helper` uses it to
        skip past an unsolicited reply and get to the one actually being
        waited on) would hang here forever: it explicitly discards
        `kernel_info_reply` messages, but that is the only reply this
        request ever produces. `get_shell_msg` with a bounded timeout is
        what jkt's own base `test_kernel_info` uses for the same request.
        """
        self.flush_channels()

        msg_id = self.kc.kernel_info()
        reply = self.kc.get_shell_msg(timeout=jkt.TIMEOUT)

        self.assertEqual(reply["msg_type"], "kernel_info_reply")
        content = reply["content"]

        self.assertEqual(content["status"], "ok")
        self.assertEqual(content["protocol_version"], "5.3")
        self.assertEqual(content["implementation"], "ggsql-jupyter")

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
        """A plot arrives as a rendered image bundle, in this build's format."""
        self.flush_channels()

        code = """
        SELECT 1 as x, 2 as y
        VISUALISE x, y
        DRAW point
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

        # A plot arrives as a rendered image, in whichever format this build can
        # produce: PNG with the raster writers and a GPU adapter, SVG otherwise.
        # Both are static bundles that need no network.
        data = execute_result["content"]["data"]
        self.assertTrue(
            "image/png" in data or "image/svg+xml" in data,
            f"expected a rendered plot, got {sorted(data)}",
        )

        # And a plain-text summary, for a frontend that renders neither.
        self.assertIn("text/plain", data)

        # The bundle must not claim a plot slot: `output_location` would route
        # it to Positron's Plots pane as well as the cell, showing it twice.
        self.assertNotIn("output_location", execute_result["content"])

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
        """Test that shutdown works.

        `setUpClass`/`tearDownClass` own one kernel shared by every test
        method in this class, so shutting *that* one down here would leave
        nothing for whatever test runs next (unittest orders methods
        alphabetically, so this would otherwise run before
        `test_status_messages`). Start a throwaway kernel instead.

        `kc.shutdown()` sends `shutdown_request` on the *control* channel
        (see `KernelClient.shutdown`'s docstring), and the reply comes back
        on the same channel — not shell, despite the original version of
        this test waiting on `get_shell_msg` and timing out here every time.
        """
        from jupyter_client.manager import start_new_kernel

        km, kc = start_new_kernel(kernel_name=self.kernel_name)
        try:
            msg_id = kc.shutdown()
            reply = kc.get_control_msg(timeout=5)

            self.assertEqual(reply["msg_type"], "shutdown_reply")
            self.assertEqual(reply["content"]["status"], "ok")
            self.assertIn("restart", reply["content"])
        finally:
            kc.stop_channels()
            km.shutdown_kernel()

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


# Restored in teardown_module. Set before install so the kernelspec below
# never touches a developer's real Jupyter data directory.
_original_jupyter_data_dir = None
_scratch_data_dir = None


# Configure kernel for testing
def setup_module():
    """Build the kernel once and install it into an isolated kernelspec."""
    import tempfile
    import json

    global _original_jupyter_data_dir, _scratch_data_dir

    # Isolate JUPYTER_DATA_DIR before anything below can install or remove a
    # kernelspec, so this suite can never clobber a developer's real "ggsql"
    # kernel — nothing in the environment has to know to set this itself.
    _original_jupyter_data_dir = os.environ.get("JUPYTER_DATA_DIR")
    _scratch_data_dir = tempfile.mkdtemp(prefix="ggsql-jupyter-data-")
    os.environ["JUPYTER_DATA_DIR"] = _scratch_data_dir

    # Build kernel (once for the whole module; individual tests no longer
    # rebuild it in setUp). Shared with test_integration.py via conftest.py.
    binary_path = build_kernel_binary()

    # Create kernel spec
    kernel_spec = {
        "argv": [binary_path, "-f", "{connection_file}"],
        "display_name": KERNEL_NAME,
        "language": "ggsql",
    }

    # Write kernel.json to temp directory
    spec_dir = Path(tempfile.mkdtemp(prefix="ggsql-kernel-"))
    with open(spec_dir / "kernel.json", "w") as f:
        json.dump(kernel_spec, f)

    # Install kernel spec (into the scratch JUPYTER_DATA_DIR set above)
    result = subprocess.run(
        [
            "jupyter",
            "kernelspec",
            "install",
            "--user",
            "--name",
            KERNEL_NAME,
            str(spec_dir),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Warning: Failed to install kernel spec: {result.stderr}")
        print("Tests may fail if kernel spec is not properly installed")


def teardown_module():
    """Cleanup kernel spec after tests and restore JUPYTER_DATA_DIR."""
    subprocess.run(
        ["jupyter", "kernelspec", "remove", "-f", KERNEL_NAME],
        capture_output=True,
    )

    if _original_jupyter_data_dir is None:
        os.environ.pop("JUPYTER_DATA_DIR", None)
    else:
        os.environ["JUPYTER_DATA_DIR"] = _original_jupyter_data_dir

    if _scratch_data_dir is not None:
        import shutil

        shutil.rmtree(_scratch_data_dir, ignore_errors=True)


if __name__ == "__main__":
    # Run setup
    setup_module()

    try:
        # Run tests
        unittest.main()
    finally:
        # Cleanup
        teardown_module()
