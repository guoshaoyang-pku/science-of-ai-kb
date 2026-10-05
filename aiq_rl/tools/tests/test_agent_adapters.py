import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))
import cc_agent
import codex_agent


FAKE_CLI = r'''#!/usr/bin/env python3
import json, os, sys, time, re, urllib.request, uuid
from pathlib import Path
args = sys.argv[1:]
is_codex = Path(sys.argv[0]).name == "codex"
Path("fake_argv.json").write_text(json.dumps(args))
sys.stdin.read()
def emit(event):
    print(json.dumps(event), flush=True)
if is_codex:
    cfg = (Path(os.environ["CODEX_HOME"]) / "config.toml").read_text()
    url = re.search(r'http://127[.]0[.]0[.]1:[0-9]+/', cfg).group(0)
    sid = args[2] if args[:2] == ["exec", "resume"] else str(uuid.uuid4())
    emit({"type": "thread.started", "thread_id": sid})
else:
    cfg = json.loads(Path(args[args.index("--mcp-config") + 1]).read_text())
    url = cfg["mcpServers"]["kb"]["args"][-1]
    flag = "--resume" if "--resume" in args else "--session-id"
    sid = args[args.index(flag) + 1] if flag in args else str(uuid.uuid4())
    emit({"type": "system", "subtype": "init", "session_id": sid})
outputs = []
for i in range(int(os.environ.get("FAKE_CALLS", "1"))):
    if not is_codex:
        emit({"type": "assistant", "message": {"id": str(i), "content": [
            {"type": "tool_use", "id": str(i), "name": "mcp__kb__python", "input": {"i": i}}]}})
    body = json.dumps({"name": "python" if is_codex else "mcp__kb__python",
                       "arguments": {"i": i}}).encode()
    with urllib.request.urlopen(urllib.request.Request(url, body), timeout=5) as r:
        outputs.append(r.read().decode())
    if is_codex:
        emit({"type": "item.completed", "item": {"type": "mcp_tool_call", "result": outputs[-1]}})
Path("fake_outputs.json").write_text(json.dumps(outputs))
time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
if os.environ.get("FAKE_CRASH") == "1":
    sys.exit(7)
if is_codex:
    emit({"type": "item.completed", "item": {"type": "agent_message", "text": "done"}})
    emit({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}})
else:
    emit({"type": "result", "result": "done", "is_error": False, "session_id": sid,
          "total_cost_usd": 0.01, "usage": {"input_tokens": 1}, "num_turns": 1})
'''

TOOLS = [{"type": "function", "function": {"name": "python", "description": "test tool",
          "parameters": {"type": "object", "properties": {"i": {"type": "integer"}}}}}]


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        binary_dir = self.root / "bin"
        binary_dir.mkdir()
        for name in ("codex", "claude"):
            binary = binary_dir / name
            binary.write_text(FAKE_CLI.replace("#!/usr/bin/env python3", "#!" + sys.executable, 1))
            binary.chmod(0o755)
        self.keys = self.root / "test_keys.json"
        cfg = {"api_key": "unused-test-value", "base_url": "http://localhost.invalid"}
        self.keys.write_text(json.dumps({"cctq": cfg, "cctq_claude": cfg}))
        self.patches = [patch.object(codex_agent, "KEYS", self.keys), patch.object(cc_agent, "KEYS", self.keys),
                        patch.dict(os.environ, {"PATH": str(binary_dir) + os.pathsep + os.environ["PATH"],
                                                "FAKE_CALLS": "1", "FAKE_SLEEP": "0", "FAKE_CRASH": "0"})]
        for p in self.patches:
            p.start()
        self.calls = []

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def handler(self, name, args):
        self.calls.append((name, args))
        return "ok"

    def codex(self, directory=None, **kwargs):
        return codex_agent.CodexSession(directory or self.root / "codex-job", "test-model", "test",
                                        TOOLS, self.handler, **kwargs)

    def claude(self, **kwargs):
        return cc_agent.run_claude_agent("test", "test", TOOLS, self.handler, model="test-model", effort="low",
                                         workdir=self.root / "claude-job", retries=0, **kwargs)

    def test_codex_unlimited_tools_and_exact_resume(self):
        os.environ.update(FAKE_CALLS="19", FAKE_SLEEP="0.2")
        observed = []
        session = self.codex(on_event=observed.append)
        try:
            result = session.turn("test", "low", "discover", max_calls=None, timeout=None)
            thread_id = session.thread_id
        finally:
            session.close()
        self.assertEqual(len(self.calls), 19)
        self.assertGreater(result["secs"], 0.1)
        self.assertEqual(len([e for e in observed if e["type"] == "tool.result"]), 19)
        session = self.codex()
        try:
            self.assertEqual(session.thread_id, thread_id)
            session.turn("continue", "low", "discover", max_calls=None, timeout=None)
        finally:
            session.close()
        args = json.loads((self.root / "codex-job/fake_argv.json").read_text())
        self.assertEqual(args[:3], ["exec", "resume", thread_id])
        self.assertNotIn("--last", args)

    def test_codex_solve_limit_and_refusal_preserved(self):
        os.environ["FAKE_CALLS"] = "5"
        session = self.codex(solve_tools={"python"}, solve_max_calls=2)
        try:
            session.turn("solve", "low", "solve", max_calls=None, timeout=None)
            self.assertEqual(len(self.calls), 2)
        finally:
            session.close()
        self.calls.clear()
        session = self.codex(directory=self.root / "refused-job")
        try:
            session.turn("solve", "low", "solve", max_calls=None, timeout=None)
            self.assertEqual(self.calls, [])
            out = json.loads((self.root / "refused-job/fake_outputs.json").read_text())
            self.assertTrue(all(s.startswith("[refused]") for s in out))
        finally:
            session.close()

    def test_claude_unlimited_durable_resume_and_non_durable_default(self):
        os.environ.update(FAKE_CALLS="19", FAKE_SLEEP="0.2")
        observed = []
        final, _, info = self.claude(max_calls=None, timeout=None, durable=True, on_event=observed.append)
        self.assertEqual(final, "done")
        self.assertEqual(len(self.calls), 19)
        self.assertEqual(info["calls"], 19)
        self.assertEqual(len([e for e in observed if e["type"] == "tool.result"]), 19)
        session_id = info["session_id"]
        self.claude(max_calls=None, timeout=None, durable=True)
        args = json.loads((self.root / "claude-job/fake_argv.json").read_text())
        self.assertEqual(args[args.index("--resume") + 1], session_id)
        self.assertNotIn("--no-session-persistence", args)
        self.assertTrue((self.root / "claude-job/claude_home").is_dir())
        before = len(self.calls)
        self.claude(max_calls=2, timeout=None)
        self.assertEqual(len(self.calls) - before, 2)
        args = json.loads((self.root / "claude-job/fake_argv.json").read_text())
        self.assertIn("--no-session-persistence", args)
        self.assertNotIn("--resume", args)
        out = json.loads((self.root / "claude-job/fake_outputs.json").read_text())
        self.assertTrue(all(s.startswith("[budget]") for s in out[2:]))

    def test_codex_finite_hard_timeout_keeps_recoverable_identity(self):
        os.environ["FAKE_SLEEP"] = "60"
        session = self.codex()
        try:
            with self.assertRaises(subprocess.TimeoutExpired):
                session.turn("test", "low", "discover", max_calls=None, timeout=1)
            self.assertIsNotNone(session.thread_id)
            state = json.loads((self.root / "codex-job/_codex_session.json").read_text())
            self.assertEqual(state["status"], "interrupted")
        finally:
            session.close()

    def test_codex_stream_and_session_survive_worker_kill(self):
        self.assert_worker_recovery("codex")

    def test_claude_stream_and_session_survive_worker_kill(self):
        self.assert_worker_recovery("claude")

    def test_stream_recovers_identity_if_checkpoint_write_was_interrupted(self):
        session = self.codex()
        try:
            session.turn("test", "low", "discover", max_calls=None, timeout=None)
            thread_id = session.thread_id
        finally:
            session.close()
        (self.root / "codex-job/_codex_session.json").write_text("{}")
        session = self.codex()
        try:
            self.assertEqual(session.thread_id, thread_id)
        finally:
            session.close()
        _, _, info = self.claude(max_calls=None, timeout=None, durable=True)
        session_id = info["session_id"]
        (self.root / "claude-job/_claude_session.json").write_text("{}")
        _, _, recovered = self.claude(max_calls=None, timeout=None, durable=True)
        self.assertEqual(recovered["session_id"], session_id)
        args = json.loads((self.root / "claude-job/fake_argv.json").read_text())
        self.assertEqual(args[args.index("--resume") + 1], session_id)

    def assert_worker_recovery(self, backend):
        directory = self.root / (backend + "-job")
        driver = self.root / "driver.py"
        if backend == "codex":
            body = 's = adapter.CodexSession(directory, "test-model", "test", [], lambda n,a: "ok")\ns.turn("test", "low", "discover", max_calls=None, timeout=None)\ns.close()\n'
            state_name, events_name, tools_name, id_key = "_codex_session.json", "events.jsonl", "_codex_tools.jsonl", "thread_id"
            module = "codex_agent"
        else:
            body = 'adapter.run_claude_agent("test", "test", [], lambda n,a: "ok", model="test-model", effort="low", workdir=directory, max_calls=None, timeout=None, durable=True, retries=0)\n'
            state_name, events_name, tools_name, id_key = "_claude_session.json", "_cc_events.jsonl", "_cc_tools.jsonl", "session_id"
            module = "cc_agent"
        driver.write_text(f'import sys\nfrom pathlib import Path\nsys.path.insert(0, {str(TOOLS_DIR)!r})\n'
                          f'import {module} as adapter\nadapter.KEYS = Path({str(self.keys)!r})\n'
                          f'directory = Path({str(directory)!r})\n' + body)
        env = {**os.environ, "FAKE_SLEEP": "60"}
        proc = subprocess.Popen([sys.executable, str(driver)], env=env, start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            until = time.monotonic() + 5
            while time.monotonic() < until:
                if (directory / tools_name).exists() and (directory / events_name).exists():
                    break
                self.assertIsNone(proc.poll(), "adapter exited before stream checkpoint")
                time.sleep(0.02)
            else:
                self.fail("stream checkpoint not saved while worker running")
            old = json.loads((directory / state_name).read_text())
            sid = old[id_key]
            time.sleep(0.08)
            self.assertIsNone(proc.poll(), "soft observation budget must not terminate worker")
            os.kill(old["pid"], 0)
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
        self.assertTrue((directory / tools_name).read_text())
        self.assertTrue((directory / events_name).read_text())
        if backend == "codex":
            session = self.codex(directory=directory)
            try:
                self.assertEqual(session.thread_id, sid)
                session.turn("resume", "low", "discover", max_calls=None, timeout=None)
            finally:
                session.close()
            args = json.loads((directory / "fake_argv.json").read_text())
            self.assertEqual(args[:3], ["exec", "resume", sid])
        else:
            _, _, info = self.claude(max_calls=None, timeout=None, durable=True)
            self.assertEqual(info["session_id"], sid)
            args = json.loads((directory / "fake_argv.json").read_text())
            self.assertEqual(args[args.index("--resume") + 1], sid)


if __name__ == "__main__":
    unittest.main()
