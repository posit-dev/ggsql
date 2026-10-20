"""
Integration tests for ggsql-jupyter kernel using jupyter_client.

These tests launch the kernel and send real Jupyter protocol messages
to verify correct behavior.
"""

import contextlib
import json
import time
import subprocess
import os
import pytest
from jupyter_client import KernelManager

# kernel_binary is defined in conftest.py, shared with test_compliance.py so
# the kernel is built once rather than once per test file.


def _launch(kernel_binary, extra_args=(), env=None):
    """Start a kernel and yield its manager. Shared by the fixtures below."""
    kernel_process = None
    km = None
    try:
        # Use KernelManager to write connection file with proper ports
        km = KernelManager()
        # We launch and kill the process ourselves below (km.start_kernel()
        # is never called), so km's own liveness bookkeeping never gets set
        # up: km.is_alive() would otherwise unconditionally report the
        # kernel dead, since has_kernel is False. Telling it we don't own
        # the kernel process makes wait_for_ready() rely purely on getting a
        # real kernel_info_reply within its timeout, which is what we want.
        km._owns_kernel = False
        km.write_connection_file()
        connection_file = km.connection_file

        # Start our kernel process directly
        kernel_process = subprocess.Popen(
            [kernel_binary, "-f", connection_file, *extra_args],
            env={**os.environ, **(env or {})},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Store the process in km for cleanup
        km._kernel_process = kernel_process

        # Wait for the kernel to actually be ready (heartbeat up) rather than
        # guessing with a fixed sleep. Opened and closed here rather than
        # left open, since callers that want a client build their own via
        # km.client() (see the `client`/`console_client` fixtures below).
        probe = km.client()
        probe.start_channels()
        try:
            probe.wait_for_ready(timeout=10)
        except RuntimeError:
            probe.stop_channels()
            stdout, stderr = (b"", b"")
            if kernel_process.poll() is not None:
                stdout, stderr = kernel_process.communicate()
            pytest.fail(
                f"Kernel failed to start:\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
            )
        else:
            probe.stop_channels()

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


# Also usable as `with _launch_kernel(binary) as km:` outside of pytest's
# fixture machinery — see TestShutdown.test_shutdown_request below, which
# needs a kernel of its own rather than the session-scoped one.
_launch_kernel = contextlib.contextmanager(_launch)


@pytest.fixture(scope="session")
def kernel_manager(kernel_binary):
    """Start one kernel manager, shared by every test in this session.

    Session-scoped so the suite doesn't pay kernel startup cost per test.
    Any test that needs to shut down or otherwise damage its kernel — see
    TestShutdown — must launch a private one instead of using this fixture.
    """
    yield from _launch(kernel_binary)


@pytest.fixture(scope="session")
def console_kernel_manager(kernel_binary):
    """A session-scoped kernel that believes it is a Positron console session.

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


class TestResultExplorer:
    """A console query result, its link, and `-- @view-result` opening it."""

    QUERY = "SELECT i AS n, 'r' || i AS label FROM range(5) t(i)"

    def _run_query(self, client):
        """Run QUERY and return the iopub messages it produced."""
        client.execute(self.QUERY, silent=False, store_history=True)
        messages = drain_iopub(client)
        reply = client.get_shell_msg(timeout=5)
        assert reply["content"]["status"] == "ok"
        return messages

    def _link_key(self, client):
        """Run QUERY in a console session and return the key its link names."""
        messages = self._run_query(client)
        order = [m["msg_type"] for m in messages]
        assert "execute_result" in order, order

        # Its own output after the table: the console renders the HTML and
        # ignores the bundle's text/plain, so a link there would never show.
        streams = [m for m in messages if m["msg_type"] == "stream"]
        assert len(streams) == 1, order
        assert order.index("execute_result") < order.index("stream")

        text = streams[0]["content"]["text"]
        prefix = f"\x1b]8;;x-ggsql-view:{client.session.session}/"
        assert text.startswith(prefix), repr(text)
        assert "Open in Data Explorer" in text
        return text[len(prefix):].split("\x1b", 1)[0]

    def _rpc(self, client, comm_id, method, params=None):
        client.session.send(
            client.shell_channel.socket,
            "comm_msg",
            {
                "comm_id": comm_id,
                "data": {
                    "jsonrpc": "2.0",
                    "id": f"{method}-1",
                    "method": method,
                    "params": params or {},
                },
            },
        )
        reply = client.get_shell_msg(timeout=5)
        drain_iopub(client)
        return reply["content"]["data"]["result"]

    def test_view_opens_the_result_in_an_explorer_tab(self, console_client):
        key = self._link_key(console_client)

        # How the extension runs it when the link is clicked.
        msg_id = console_client.execute(f"-- @view-result {key}", silent=True)
        messages = drain_iopub(console_client)
        reply = console_client.get_shell_msg(timeout=5)
        assert reply["content"]["status"] == "ok", reply["content"]

        opens = [m for m in messages if m["msg_type"] == "comm_open"]
        assert len(opens) == 1, [m["msg_type"] for m in messages]
        content = opens[0]["content"]
        assert content["target_name"] == "positron.dataExplorer"
        # Not inline: Positron opens a tab for it straight away.
        assert content["data"]["inline_only"] is False
        assert opens[0]["parent_header"]["msg_id"] == msg_id

        comm_id = content["comm_id"]
        state = self._rpc(console_client, comm_id, "get_state")
        assert state["display_name"].startswith("Query [")
        assert state["table_shape"] == {"num_rows": 5, "num_columns": 2}

        values = self._rpc(
            console_client,
            comm_id,
            "get_data_values",
            {
                "columns": [
                    {"column_index": 1, "spec": {"first_index": 3, "last_index": 4}}
                ]
            },
        )
        assert values["columns"] == [["r3", "r4"]]

    def _open(self, client):
        """Open QUERY's result in an explorer and return the comm id."""
        key = self._link_key(client)
        client.execute(f"-- @view-result {key}", silent=True)
        messages = drain_iopub(client)
        client.get_shell_msg(timeout=5)
        return next(
            m["content"]["comm_id"] for m in messages if m["msg_type"] == "comm_open"
        )

    def test_filter_sort_and_convert_to_code(self, console_client):
        comm_id = self._open(console_client)

        state = self._rpc(console_client, comm_id, "get_state")
        features = state["supported_features"]
        for feature in ("set_row_filters", "set_sort_columns", "convert_to_code"):
            assert features[feature]["support_status"] == "supported", feature

        n_column = {"column_index": 0}
        reply = self._rpc(
            console_client,
            comm_id,
            "set_row_filters",
            {
                "filters": [
                    {
                        "filter_id": "f1",
                        "filter_type": "compare",
                        "column_schema": n_column,
                        "condition": "and",
                        "params": {"op": ">=", "value": "2"},
                    }
                ]
            },
        )
        assert reply == {"selected_num_rows": 3, "had_errors": False}

        self._rpc(
            console_client,
            comm_id,
            "set_sort_columns",
            {"sort_keys": [{"column_index": 0, "ascending": False}]},
        )
        values = self._rpc(
            console_client,
            comm_id,
            "get_data_values",
            {
                "columns": [
                    {"column_index": 1, "spec": {"first_index": 0, "last_index": 2}}
                ]
            },
        )
        assert values["columns"] == [["r4", "r3", "r2"]]

        code = self._rpc(
            console_client,
            comm_id,
            "convert_to_code",
            {
                "column_filters": [],
                "row_filters": state["row_filters"],
                "sort_keys": [{"column_index": 0, "ascending": False}],
                "code_syntax_name": {"code_syntax_name": "SQL"},
            },
        )
        assert f"    {self.QUERY}" in code["converted_code"]
        assert 'ORDER BY "n" DESC' in code["converted_code"]

    def test_a_bad_request_is_a_json_rpc_error(self, console_client):
        comm_id = self._open(console_client)
        console_client.session.send(
            console_client.shell_channel.socket,
            "comm_msg",
            {
                "comm_id": comm_id,
                "data": {
                    "jsonrpc": "2.0",
                    "id": "export-1",
                    "method": "export_data_selection",
                    "params": {
                        "format": "csv",
                        "selection": {"kind": "nonsense", "selection": {}},
                    },
                },
            },
        )
        reply = console_client.get_shell_msg(timeout=5)
        drain_iopub(console_client)
        data = reply["content"]["data"]
        assert "error" in data and "result" not in data, data

    def test_a_second_click_focuses_the_open_explorer(self, console_client):
        # Positron opens a UI comm on every console session; focusing an
        # editor is an event on it.
        ui = "test-ui-comm"
        console_client.session.send(
            console_client.shell_channel.socket,
            "comm_open",
            {"comm_id": ui, "target_name": "positron.ui", "data": {}},
        )
        drain_iopub(console_client)
        try:
            key = self._link_key(console_client)
            console_client.execute(f"-- @view-result {key}", silent=True)
            first = drain_iopub(console_client)
            console_client.get_shell_msg(timeout=5)
            comm_id = next(
                m["content"]["comm_id"] for m in first if m["msg_type"] == "comm_open"
            )

            console_client.execute(f"-- @view-result {key}", silent=True)
            second = drain_iopub(console_client)
            reply = console_client.get_shell_msg(timeout=5)
            assert reply["content"]["status"] == "ok", reply["content"]

            types = [m["msg_type"] for m in second]
            assert "comm_open" not in types, types
            events = [
                m["content"]
                for m in second
                if m["msg_type"] == "comm_msg" and m["content"]["comm_id"] == ui
            ]
            assert len(events) == 1, types
            assert events[0]["data"]["method"] == "open_editor"
            params = events[0]["data"]["params"]
            assert params["kind"] == "uri"
            assert params["file"] == f"positron-data-explorer:positron-data-explorer-{comm_id}"
        finally:
            console_client.session.send(
                console_client.shell_channel.socket,
                "comm_close",
                {"comm_id": ui, "data": {}},
            )
            drain_iopub(console_client)

    def test_a_result_that_is_gone_is_an_error(self, console_client):
        console_client.execute("-- @view-result deadbeef", silent=True)
        drain_iopub(console_client)
        reply = console_client.get_shell_msg(timeout=5)
        assert reply["content"]["status"] == "error"
        assert "no longer available" in reply["content"]["evalue"]

    def test_only_a_console_gets_a_link(self, client):
        messages = self._run_query(client)
        assert not [m for m in messages if m["msg_type"] == "stream"]


class TestViewDirective:
    """`-- @view` sending a query's result to the data explorer."""

    CODE = "-- @view: Small numbers\nSELECT i AS n FROM range(4) t(i)"

    def _run(self, client, code, silent=False):
        client.execute(code, silent=silent, store_history=not silent)
        messages = drain_iopub(client)
        reply = client.get_shell_msg(timeout=5)
        return messages, reply["content"]

    def test_the_result_opens_in_an_explorer_tab(self, console_client):
        messages, reply = self._run(console_client, self.CODE)
        assert reply["status"] == "ok", reply
        types = [m["msg_type"] for m in messages]

        # The explorer, and the table in the console as for any query.
        opens = [m for m in messages if m["msg_type"] == "comm_open"]
        assert len(opens) == 1, types
        data = opens[0]["content"]["data"]
        assert data == {"title": "Small numbers", "inline_only": False}
        assert types.count("execute_result") == 1, types

        # And the link under the table, to reopen the tab once it is closed.
        streams = [m for m in messages if m["msg_type"] == "stream"]
        assert len(streams) == 1, types
        assert "x-ggsql-view:" in streams[0]["content"]["text"]
        assert types.index("execute_result") < types.index("stream"), types

    def test_a_silent_execution_opens_it_without_output(self, console_client):
        messages, reply = self._run(console_client, self.CODE, silent=True)
        assert reply["status"] == "ok", reply
        types = [m["msg_type"] for m in messages]
        assert types.count("comm_open") == 1, types
        assert "stream" not in types and "execute_result" not in types, types

    def test_a_plot_query_is_an_error(self, console_client):
        messages, reply = self._run(
            console_client,
            "-- @view\nSELECT 1 AS x, 2 AS y VISUALISE x, y DRAW point",
        )
        assert reply["status"] == "error"
        assert "not a plot" in reply["evalue"]
        assert "comm_open" not in [m["msg_type"] for m in messages]

    def test_outside_positron_the_table_shows_inline(self, client):
        messages, reply = self._run(client, self.CODE)
        assert reply["status"] == "ok", reply
        types = [m["msg_type"] for m in messages]
        assert "execute_result" in types and "comm_open" not in types, types
        notes = [
            m["content"]["text"]
            for m in messages
            if m["msg_type"] == "stream" and m["content"]["name"] == "stderr"
        ]
        assert notes and "needs Positron" in notes[0], types


class TestConsoleRows:
    """`GGSQL_CONSOLE_ROWS`, which the extension sets from `ggsql.consoleRows`."""

    def _table(self, client, n):
        client.execute(f"SELECT i FROM range({n}) t(i)")
        messages = drain_iopub(client)
        client.get_shell_msg(timeout=5)
        result = next(m for m in messages if m["msg_type"] == "execute_result")
        return result["content"]["data"]["text/html"]

    def test_the_environment_sets_the_rows_a_console_shows(self, kernel_binary):
        with _launch_kernel(
            kernel_binary,
            ["--session-mode", "console"],
            env={"GGSQL_CONSOLE_ROWS": "3"},
        ) as km:
            kc = km.client()
            kc.start_channels()
            try:
                kc.wait_for_ready(timeout=10)
                # Up to twice the setting, the whole result.
                assert self._table(kc, 6).count("<tr>") == 1 + 6
                # Past it, the first 3 and a count of the rest.
                html = self._table(kc, 7)
                assert html.count("<tr>") == 1 + 3 + 1, html
                assert "... 4 more rows" in html
            finally:
                kc.stop_channels()


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

    def test_shutdown_request(self, kernel_binary):
        """Test that kernel responds to shutdown_request.

        Launches its own kernel rather than taking the session-scoped
        `kernel_manager` fixture: this test's whole point is to shut its
        kernel down, which would take every other test in this module down
        with it if it were the shared one.

        The reply comes back on the *control* channel — `kc.shutdown()`
        sends `shutdown_request` there, per `KernelClient.shutdown`'s
        docstring — not shell.
        """
        with _launch_kernel(kernel_binary) as km:
            kc = km.client()
            kc.start_channels()
            try:
                kc.wait_for_ready(timeout=10)

                msg_id = kc.shutdown()
                reply = kc.get_control_msg(timeout=10)

                assert reply["msg_type"] == "shutdown_reply"
                assert reply["content"]["status"] == "ok"
                assert "restart" in reply["content"]
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
