#!/usr/bin/env python3
"""
Textual-based installer for Synthesia2MIDI.

This provides a guided, non-technical setup experience that installs
all required dependencies and keeps the terminal open on failure.
"""
from __future__ import annotations

import asyncio
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import webbrowser
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, OptionList, ProgressBar, RichLog, Static
from textual.widgets._option_list import Option


PALETTE = {
    "text": "#E8E3D5",
    "dim": "#7B7F87",
    "accent": "#F6C453",
    "accent_soft": "#F2A65A",
    "border": "#3C414B",
    "panel": "#2B2F36",
    "code_bg": "#1E232A",
    "success": "#7DD3A5",
    "error": "#F97066",
}


@dataclass
class Step:
    key: str
    title: str
    description: str
    action: Callable[[], Awaitable[None]]


@dataclass
class FixOption:
    label: str
    action: Callable[[], Awaitable[None]]
    post_hint: Optional[str] = None


class StepFailure(Exception):
    def __init__(
        self,
        title: str,
        message: str,
        instructions: List[str],
        help_url: Optional[str] = None,
        fix_options: Optional[List[FixOption]] = None,
    ):
        super().__init__(message)
        self.title = title
        self.message = message
        self.instructions = instructions
        self.help_url = help_url
        self.fix_options = fix_options or []


class InstallerApp(App):
    CSS = f"""
    Screen {{
        background: {PALETTE['code_bg']};
        color: {PALETTE['text']};
    }}

    #header {{
        height: 3;
        background: {PALETTE['panel']};
        color: {PALETTE['accent_soft']};
        content-align: center middle;
        border-bottom: tall {PALETTE['border']};
        text-style: bold;
    }}

    #footer {{
        height: 3;
        background: {PALETTE['panel']};
        color: {PALETTE['dim']};
        content-align: center middle;
        border-top: tall {PALETTE['border']};
    }}

    #content {{
        height: 1fr;
    }}

    #steps {{
        width: 30;
        border-right: tall {PALETTE['border']};
        padding: 1 1;
        background: {PALETTE['panel']};
    }}

    #main {{
        padding: 1 2;
    }}

    #progress_label,
    #step_label {{
        color: {PALETTE['dim']};
        text-style: bold;
        height: 1;
    }}

    #progress_bar,
    #step_bar {{
        height: 1;
        margin: 0 0 1 0;
    }}

    #main_title {{
        color: {PALETTE['accent']};
        text-style: bold;
    }}

    #main_body {{
        color: {PALETTE['text']};
    }}

    #main_hint {{
        color: {PALETTE['dim']};
    }}

    #actions {{
        height: 3;
        align: center middle;
    }}

    #open_link_button {{
        background: {PALETTE['accent']};
        color: {PALETTE['code_bg']};
        border: round {PALETTE['accent_soft']};
        text-style: bold;
        padding: 0 2;
    }}

    #fix_options_label {{
        color: {PALETTE['dim']};
        text-style: bold;
        height: 1;
        margin-top: 1;
    }}

    #fix_options {{
        height: 6;
        border: tall {PALETTE['border']};
        background: {PALETTE['panel']};
        color: {PALETTE['text']};
        padding: 0 1;
    }}

    #details {{
        border: tall {PALETTE['border']};
        height: 1fr;
        padding: 1 1;
        background: {PALETTE['panel']};
        color: {PALETTE['dim']};
    }}
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "retry", "Retry"),
        ("i", "reinstall", "Reinstall"),
        ("o", "open_link", "Open link"),
        ("s", "show_next_failure", "Next failure"),
        ("d", "toggle_details", "Details"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.steps: List[Step] = []
        self.step_state: dict[str, str] = {}
        self.failed_index: Optional[int] = None
        self.failure_url: Optional[str] = None
        self.fix_option_map: dict[str, FixOption] = {}
        self.fix_running = False
        self.running = False
        self.force_reinstall = False
        self.test_mode = os.environ.get("S2M_TEST_MODE", "").strip().lower() in {"1", "true", "yes"}
        self.test_fail_step = os.environ.get("S2M_TEST_FAIL_STEP", "").strip().lower()
        self.test_sequence = os.environ.get("S2M_TEST_SEQUENCE", "").strip().lower() in {"1", "true", "yes"}
        self.test_sequence_steps = ["env", "deps", "ffmpeg", "rust", "build"]
        if self.test_mode and self.test_sequence:
            if self.test_fail_step in self.test_sequence_steps:
                self.test_sequence_index = self.test_sequence_steps.index(self.test_fail_step)
            else:
                self.test_sequence_index = 0
                self.test_fail_step = self.test_sequence_steps[0]
        else:
            self.test_sequence_index = 0
        self.repo_root = Path(__file__).resolve().parent.parent
        self.venv_python = self._resolve_venv_python()
        self.log_file = self.repo_root / "logs" / "installer_tui.log"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.details_visible = False
        self._working_timer = None
        self._working_frames = (".", "..", "...")
        self._working_index = 0
        self._working_prefix = "Working"

        self.header = Static(self._build_brand_banner(), id="header")
        self.steps_panel = Static(id="steps")
        self.overall_label = Static("Overall progress", id="progress_label")
        self.overall_bar = ProgressBar(total=1, show_eta=False, id="progress_bar")
        self.step_label = Static("Current step", id="step_label")
        self.step_bar = ProgressBar(total=1, show_eta=False, id="step_bar")
        self.main_title = Static(id="main_title")
        self.main_body = Static(id="main_body")
        self.main_hint = Static(id="main_hint")
        self.open_link_button = Button("Open help page", id="open_link_button")
        self.fix_options_label = Static("Fix options (use arrows + Enter):", id="fix_options_label")
        self.fix_options_list = OptionList(id="fix_options")
        self.details = RichLog(id="details", wrap=True, highlight=False)
        self.footer = Static(self._footer_text(), id="footer")

    def compose(self) -> ComposeResult:
        yield self.header
        with Horizontal(id="content"):
            yield self.steps_panel
            with Vertical(id="main"):
                yield self.overall_label
                yield self.overall_bar
                yield self.step_label
                yield self.step_bar
                yield self.main_title
                yield self.main_body
                yield self.main_hint
                with Horizontal(id="actions"):
                    yield self.open_link_button
                yield self.fix_options_label
                yield self.fix_options_list
                yield self.details
        yield self.footer

    async def on_mount(self) -> None:
        self.details.display = False
        self.open_link_button.display = False
        self.fix_options_label.display = False
        self.fix_options_list.display = False
        self._build_steps()
        self._render_steps()
        self._set_overall_progress(0)
        self.call_after_refresh(self._start_steps)

    def _build_brand_banner(self) -> Text:
        banner = Text("\nSYNTHESIA", style=f"bold {PALETTE['accent_soft']}")
        banner.append("2", style="bold #8B5CF6")
        banner.append("MIDI", style=f"bold {PALETTE['accent_soft']}")
        if self.test_mode:
            banner.append(" [TEST MODE]", style=PALETTE["dim"])
        return banner

    def _start_steps(self) -> None:
        asyncio.create_task(self._run_steps(0))

    def _footer_text(self) -> str:
        base = "R: Retry    I: Reinstall    O: Open Link    D: Details    Q: Quit"
        if self.test_mode:
            return "S: Next Failure    " + base
        return base

    def _update_footer(self) -> None:
        self.footer.update(self._footer_text())

    def action_toggle_details(self) -> None:
        self.details_visible = not self.details_visible
        self.details.display = self.details_visible
        self._update_footer()

    async def action_retry(self) -> None:
        if self.running:
            return
        if self.failed_index is None:
            return
        start = self.failed_index
        self.failed_index = None
        self.failure_url = None
        self.force_reinstall = False
        self.open_link_button.display = False
        self._clear_fix_options()
        await self._run_steps(start)

    async def action_reinstall(self) -> None:
        if self.running:
            return
        self.force_reinstall = True
        self.failed_index = None
        self.failure_url = None
        self.step_state = {step.key: "pending" for step in self.steps}
        self._render_steps()
        self.open_link_button.display = False
        self._clear_fix_options()
        self._set_main(
            "Reinstalling",
            "We are reinstalling all dependencies from scratch.",
            "This can take several minutes.",
        )
        await self._run_steps(0)

    async def action_show_next_failure(self) -> None:
        if not self.test_mode:
            return
        if self.running or self.fix_running:
            return
        next_step = self._next_test_failure_step()
        self.test_fail_step = next_step
        self.failed_index = None
        self.failure_url = None
        self.force_reinstall = False
        self.step_state = {step.key: "pending" for step in self.steps}
        self._render_steps()
        self.open_link_button.display = False
        self._clear_fix_options()
        self._set_main(
            "Showing next test failure",
            f"Simulating failure for step: {next_step}",
            "Working...",
        )
        await self._run_steps(0)

    def action_open_link(self) -> None:
        if not self.failure_url:
            return
        self._log(f"Opening help page: {self.failure_url}")
        try:
            opened = webbrowser.open(self.failure_url)
        except Exception as exc:
            self._log(f"Failed to open browser: {exc}")
            opened = False
        if opened:
            self.main_hint.update("We opened the help page in your browser.")
        else:
            self.main_hint.update("We could not open the browser. Please copy the link above.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "open_link_button":
            self.action_open_link()

    async def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if self.running or self.fix_running:
            return
        option = self.fix_option_map.get(event.option.id)
        if not option:
            return
        self.fix_running = True
        self._start_working_indicator("Working on that now")
        try:
            await option.action()
            if option.post_hint and not self.running:
                self.main_hint.update(option.post_hint)
        finally:
            self._stop_working_indicator()
            self.fix_running = False

    def _build_steps(self) -> None:
        self.steps = [
            Step("env", "Check environment", "Verify and prepare the setup environment.", self.step_env),
            Step("deps", "Install Python packages", "Install the core Python packages for the app.", self.step_python_deps),
            Step("ffmpeg", "Install FFmpeg", "Install FFmpeg so videos can be processed.", self.step_ffmpeg),
            Step("rust", "Install Rust", "Install Rust (required for MIDI Touch-Up Editor).", self.step_rust),
            Step("build", "Build MIDI Touch-Up Editor", "Build the Rust tool used for MIDI touch-up.", self.step_build_rust),
            Step("done", "Finish", "Setup complete.", self.step_done),
        ]
        self.step_state = {step.key: "pending" for step in self.steps}
        self.overall_bar.total = len(self.steps)

    def _render_steps(self) -> None:
        text = Text()
        for step in self.steps:
            state = self.step_state.get(step.key, "pending")
            if state == "done":
                prefix = "[OK]"
                style = f"bold {PALETTE['success']}"
            elif state == "running":
                prefix = "[>>]"
                style = f"bold {PALETTE['accent']}"
            elif state == "failed":
                prefix = "[!!]"
                style = f"bold {PALETTE['error']}"
            else:
                prefix = "[  ]"
                style = PALETTE["dim"]
            line = Text(f"{prefix} {step.title}\n", style=style)
            text.append(line)
        self.steps_panel.update(text)

    def _set_main(self, title: str, body: str, hint: str = "") -> None:
        self.main_title.update(title)
        self.main_body.update(body)
        self.main_hint.update(hint)

    def _set_overall_progress(self, progress: float) -> None:
        self.overall_bar.total = max(len(self.steps), 1)
        self.overall_bar.progress = progress

    def _set_step_progress(self, total: float, progress: float = 0, label: str = "") -> None:
        self.step_bar.total = max(total, 1)
        self.step_bar.progress = progress
        if label:
            self.step_label.update(label)

    def _advance_step_progress(self, label: str = "") -> None:
        self.step_bar.advance(1)
        if label:
            self.step_label.update(label)

    def _update_step_progress(self, progress: float) -> None:
        self.step_bar.progress = progress

    def _clear_fix_options(self) -> None:
        self.fix_option_map.clear()
        self.fix_options_list.clear_options()
        self.fix_options_label.display = False
        self.fix_options_list.display = False

    def _set_fix_options(self, options: List[FixOption]) -> None:
        self.fix_option_map.clear()
        self.fix_options_list.clear_options()
        if not options:
            self.fix_options_label.display = False
            self.fix_options_list.display = False
            return
        for idx, option in enumerate(options):
            option_id = f"fix_{idx}"
            self.fix_option_map[option_id] = option
            self.fix_options_list.add_option(Option(option.label, id=option_id))
        self.fix_options_label.display = True
        self.fix_options_list.display = True
        self.fix_options_list.highlighted = 0
        self.fix_options_list.focus()

    def _build_default_fix_options(self, failure: StepFailure) -> List[FixOption]:
        options: List[FixOption] = []
        options.extend(failure.fix_options)
        options.append(FixOption("Retry this step", self._fix_retry))
        options.append(FixOption("Reinstall everything", self._fix_reinstall))
        if failure.help_url:
            options.append(FixOption("Open help page", self._fix_open_link))
        return options

    async def _simulate_progress(self, label: str, steps: int = 6, delay: float = 0.15) -> None:
        self._set_step_progress(steps, 0, label)
        for _ in range(steps):
            await asyncio.sleep(delay)
            self._advance_step_progress(label)

    def _test_should_fail(self, key: str) -> bool:
        return self.test_mode and self.test_fail_step == key

    def _next_test_failure_step(self) -> str:
        if not self.test_sequence_steps:
            return "env"
        self.test_sequence_index = (self.test_sequence_index + 1) % len(self.test_sequence_steps)
        return self.test_sequence_steps[self.test_sequence_index]

    def _is_pip_cache_permission_error(self, output: List[str]) -> bool:
        joined = "\n".join(output).lower()
        if "permission denied" not in joined and "access is denied" not in joined:
            return False
        return "pip\\cache" in joined or "pip/cache" in joined or "pip cache" in joined

    async def _fix_retry(self) -> None:
        await self.action_retry()

    async def _fix_reinstall(self) -> None:
        await self.action_reinstall()

    async def _fix_open_link(self) -> None:
        self.action_open_link()

    async def _fix_install_homebrew(self) -> None:
        self._set_step_progress(100, 0, "Installing Homebrew")
        self.main_hint.update("Installing Homebrew. This may ask for your password.")
        env = os.environ.copy()
        env["NONINTERACTIVE"] = "1"
        command = "/bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        code = await self._run_command(
            ["/bin/bash", "-c", command],
            cwd=self.repo_root,
            env=env,
            log_label="Install Homebrew",
        )
        if code == 0 and shutil.which("brew"):
            self.main_hint.update("Homebrew installed. Press R to retry.")
        else:
            self.main_hint.update("Homebrew install failed. Press D for details or Open help page.")

    async def _fix_clear_pip_cache_and_retry(self) -> None:
        self.main_hint.update("Clearing pip cache for all users...")
        await self._clear_pip_cache_all_users()
        self.main_hint.update("Cache cleared. Retrying this step...")
        await self.action_retry()

    async def _clear_pip_cache_all_users(self) -> None:
        # Clear current user's cache via pip, then delete common cache paths for all users.
        try:
            await self._run_command(
                [str(self.venv_python), "-m", "pip", "cache", "purge"],
                cwd=self.repo_root,
                log_label="Clear pip cache (current user)",
            )
        except Exception as exc:
            self._log(f"Failed to clear pip cache via pip: {exc}")

        system = platform.system().lower()
        candidate_dirs: List[Path] = []

        if system.startswith("win"):
            users_root = Path("C:/Users")
            if users_root.exists():
                for user_dir in users_root.iterdir():
                    cache_dir = user_dir / "AppData" / "Local" / "pip" / "Cache"
                    cache_dir_lower = user_dir / "AppData" / "Local" / "pip" / "cache"
                    candidate_dirs.extend([cache_dir, cache_dir_lower])
        elif system == "darwin":
            users_root = Path("/Users")
            if users_root.exists():
                for user_dir in users_root.iterdir():
                    candidate_dirs.append(user_dir / "Library" / "Caches" / "pip")
        else:
            users_root = Path("/home")
            if users_root.exists():
                for user_dir in users_root.iterdir():
                    candidate_dirs.append(user_dir / ".cache" / "pip")

        for path in candidate_dirs:
            if not path.exists():
                continue
            try:
                shutil.rmtree(path)
                self._log(f"Cleared pip cache: {path}")
            except Exception as exc:
                self._log(f"Could not clear cache {path}: {exc}")

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        self.details.write(line)

    async def _run_blocking(self, func: Callable[..., None], *args) -> None:
        await asyncio.to_thread(func, *args)

    def _start_working_indicator(self, prefix: str = "Working") -> None:
        self._stop_working_indicator()
        self._working_prefix = prefix
        self._working_index = 0
        self.main_hint.update(f"{self._working_prefix}{self._working_frames[self._working_index]}")
        self._working_timer = self.set_interval(0.35, self._tick_working_indicator)

    def _tick_working_indicator(self) -> None:
        self._working_index = (self._working_index + 1) % len(self._working_frames)
        self.main_hint.update(f"{self._working_prefix}{self._working_frames[self._working_index]}")

    def _stop_working_indicator(self) -> None:
        if self._working_timer is not None:
            self._working_timer.stop()
            self._working_timer = None

    async def _run_steps(self, start_index: int) -> None:
        self.running = True
        for idx, step in enumerate(self.steps[start_index:], start=start_index):
            self.failure_url = None
            self.step_state[step.key] = "running"
            self._render_steps()
            self._set_main(step.title, step.description, "")
            self._start_working_indicator("Working")
            self.open_link_button.display = False
            self._clear_fix_options()
            self._set_overall_progress(idx)
            self._set_step_progress(1, 0, "Starting...")
            self._log(f"Starting step: {step.title}")
            try:
                await step.action()
            except StepFailure as exc:
                self._stop_working_indicator()
                self.step_state[step.key] = "failed"
                self.failed_index = idx
                self.failure_url = exc.help_url
                self._render_steps()
                instructions = "\n".join(exc.instructions)
                if exc.help_url:
                    link_line = f"[link=\"{exc.help_url}\"]{exc.help_url}[/link]"
                    instructions = f"{instructions}\nOpen this page in your browser: {link_line}"
                    hint = "Use the Fix Options below (arrows + Enter), or press R, I, or O."
                    self.open_link_button.display = True
                else:
                    hint = "Use the Fix Options below (arrows + Enter), or press R or I. Press D for details."
                self._set_fix_options(self._build_default_fix_options(exc))
                self._set_main(
                    exc.title,
                    f"{exc.message}\n\nWhat to do next:\n{instructions}",
                    hint,
                )
                self._log(f"Step failed: {exc.title} - {exc.message}")
                self.running = False
                return
            except Exception as exc:
                self._stop_working_indicator()
                self.step_state[step.key] = "failed"
                self.failed_index = idx
                self.failure_url = None
                self._render_steps()
                self._set_main(
                    "Setup failed",
                    f"Something went wrong: {exc}",
                    f"See the log file for details: {self.log_file}",
                )
                self._log(f"Unexpected error: {exc}")
                self.running = False
                return
            self._stop_working_indicator()
            self.step_state[step.key] = "done"
            self._render_steps()
            self._set_overall_progress(idx + 1)
        self.failed_index = None
        self.running = False

    async def _run_command(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        env: Optional[dict] = None,
        log_label: Optional[str] = None,
    ) -> int:
        if log_label:
            self._log(f"Running: {log_label}")
        else:
            self._log("Running command: " + " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(cwd) if cwd else None,
            env=env,
        )
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode(errors="replace").rstrip()
            if text:
                self._log(text)
        return await proc.wait()

    async def _run_command_collect(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        env: Optional[dict] = None,
        log_label: Optional[str] = None,
    ) -> tuple[int, List[str]]:
        if log_label:
            self._log(f"Running: {log_label}")
        else:
            self._log("Running command: " + " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(cwd) if cwd else None,
            env=env,
        )
        assert proc.stdout is not None
        output: List[str] = []
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode(errors="replace").rstrip()
            if text:
                output.append(text)
                self._log(text)
        return await proc.wait(), output

    def _resolve_venv_python(self) -> Path:
        if platform.system().lower().startswith("win"):
            return self.repo_root / ".venv" / "Scripts" / "python.exe"
        return self.repo_root / ".venv" / "bin" / "python"

    def _ensure_venv_sync(self) -> None:
        if self.venv_python.exists():
            return
        self._log("Virtual environment missing. Creating .venv...")
        cmd = [sys.executable, "-m", "venv", str(self.repo_root / ".venv")]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise StepFailure(
                "Python setup failed",
                "We could not create the Python environment.",
                [
                    "Close this window.",
                    "Move this folder to a simple location (like your Desktop) and run setup again.",
                    "If it still fails, reinstall Python and make sure the checkbox 'Add Python to PATH' is checked.",
                ],
                help_url="https://www.python.org/downloads/",
            )
        self.venv_python = self._resolve_venv_python()

    async def step_env(self) -> None:
        if self.test_mode:
            await self._simulate_progress("Checking environment", steps=2)
            if self._test_should_fail("env"):
                raise StepFailure(
                    "Python setup failed",
                    "We could not create the Python environment.",
                    [
                        "Close this window.",
                        "Move this folder to a simple location (like your Desktop) and run setup again.",
                        "If it still fails, reinstall Python and make sure the checkbox 'Add Python to PATH' is checked.",
                    ],
                    help_url="https://www.python.org/downloads/",
                )
            self._set_step_progress(2, 2, "Environment ready (test mode)")
            self._log("Test mode: skipped environment checks.")
            return
        self._set_step_progress(2, 0, "Checking environment")
        await self._run_blocking(self._ensure_venv_sync)
        self._advance_step_progress("Validating Python")
        if not self.venv_python.exists():
            raise StepFailure(
                "Python setup failed",
                "The setup environment could not be created.",
                [
                    "Close this window.",
                    "Move this folder to a simple location (like your Desktop) and run setup again.",
                    "If it still fails, reinstall Python and make sure the checkbox 'Add Python to PATH' is checked.",
                ],
                help_url="https://www.python.org/downloads/",
            )
        self._advance_step_progress("Environment ready")
        self._log(f"Python version: {sys.version}")
        self._log(f"Using Python: {self.venv_python}")

    async def step_python_deps(self) -> None:
        if self.test_mode:
            await self._simulate_progress("Installing required packages", steps=3)
            if self._test_should_fail("deps"):
                raise StepFailure(
                    "Python packages failed",
                    "We could not install the required Python packages.",
                    [
                        "Press R to try again.",
                        "If it still fails, press I to reinstall everything.",
                        "If you use antivirus or company security, allow this folder.",
                        "If it fails again, press D to show details.",
                        "Open the file logs/installer_tui.log (in this folder) and share it for help.",
                    ],
                )
            self._set_step_progress(3, 3, "Packages installed (test mode)")
            self._log("Test mode: skipped Python package installs.")
            return
        self._set_step_progress(3, 0, "Updating pip")
        requirements = self.repo_root / "synthesia2midi" / "requirements.txt"
        if not requirements.exists():
            raise StepFailure(
                "Missing requirements file",
                "We could not find the file that lists required Python packages.",
                [
                    "Delete this folder.",
                    "Download the app again and extract all files.",
                    "Run setup again.",
                ],
            )

        commands = [
            ("Updating pip", [str(self.venv_python), "-m", "pip", "install", "--upgrade", "pip"]),
            ("Installing required packages", [str(self.venv_python), "-m", "pip", "install", "-r", str(requirements)]),
            ("Installing yt-dlp", [str(self.venv_python), "-m", "pip", "install", "--upgrade", "yt-dlp"]),
        ]
        for label, cmd in commands:
            self._set_step_progress(self.step_bar.total, self.step_bar.progress, label)
            if "install -r" in " ".join(cmd):
                code, output = await self._run_command_collect(cmd, cwd=self.repo_root)
            else:
                code = await self._run_command(cmd, cwd=self.repo_root)
                output = []
            if code != 0:
                fix_options: List[FixOption] = []
                instructions = [
                    "Press R to try again.",
                    "If it still fails, press I to reinstall everything.",
                    "If you use antivirus or company security, allow this folder.",
                    "If it fails again, press D to show details.",
                    "Open the file logs/installer_tui.log (in this folder) and share it for help.",
                ]
                if self._is_pip_cache_permission_error(output):
                    fix_options.append(
                        FixOption(
                            "Clear pip cache for all users and retry",
                            self._fix_clear_pip_cache_and_retry,
                        )
                    )
                    instructions = [
                        "Select 'Clear pip cache for all users and retry' below.",
                        "If it still fails, press I to reinstall everything.",
                        "If you use antivirus or company security, allow this folder.",
                        "If it fails again, press D to show details.",
                        "Open the file logs/installer_tui.log (in this folder) and share it for help.",
                    ]
                raise StepFailure(
                    "Python packages failed",
                    "We could not install the required Python packages.",
                    instructions,
                    fix_options=fix_options,
                )
            self._advance_step_progress(label)

    async def step_ffmpeg(self) -> None:
        if self.test_mode:
            await self._simulate_progress("Installing FFmpeg", steps=5)
            if self._test_should_fail("ffmpeg"):
                system = platform.system().lower()
                if system.startswith("win"):
                    target_dir = self.repo_root / "synthesia2midi" / "ffmpeg"
                    raise StepFailure(
                        "FFmpeg download failed",
                        "We downloaded FFmpeg but could not find the executable.",
                        [
                            "Click the Open help page button.",
                            "Download the file named ffmpeg-release-essentials.zip.",
                            "Unzip it and copy ffmpeg.exe into:",
                            f"{target_dir}",
                            "Press R to try again.",
                        ],
                        help_url="https://www.gyan.dev/ffmpeg/builds/",
                    )
                if system == "darwin":
                    raise StepFailure(
                        "Homebrew is not installed",
                        "We need Homebrew to install FFmpeg on macOS.",
                        [
                            "Click the Open help page button.",
                            "Follow the install steps on that page.",
                            "Then press R to try again.",
                        ],
                        help_url="https://brew.sh",
                        fix_options=[FixOption("Install Homebrew for me", self._fix_install_homebrew)],
                    )
                raise StepFailure(
                    "FFmpeg missing",
                    "We could not install FFmpeg on this system automatically.",
                    [
                        "Click the Open help page button.",
                        "Follow the install steps for your system.",
                        "Then press R to try again.",
                    ],
                    help_url="https://ffmpeg.org/download.html",
                )
            self._set_step_progress(100, 100, "FFmpeg ready (test mode)")
            self._log("Test mode: skipped FFmpeg install.")
            return
        self._set_step_progress(100, 0, "Checking FFmpeg")
        if self._find_ffmpeg() and not self.force_reinstall:
            self._log("FFmpeg already installed.")
            self._set_step_progress(100, 100, "FFmpeg already installed")
            return
        if platform.system().lower().startswith("win"):
            await self._install_ffmpeg_windows()
        elif platform.system().lower() == "darwin":
            await self._install_ffmpeg_macos()
        else:
            raise StepFailure(
                "FFmpeg missing",
                "We could not install FFmpeg on this system automatically.",
                [
                    "Click the Open help page button.",
                    "Follow the install steps for your system.",
                    "Then press R to try again.",
                ],
                help_url="https://ffmpeg.org/download.html",
            )
        if not self._find_ffmpeg():
            system = platform.system().lower()
            if system.startswith("win"):
                target_dir = self.repo_root / "synthesia2midi" / "ffmpeg"
                raise StepFailure(
                    "FFmpeg install failed",
                    "FFmpeg did not install correctly.",
                    [
                        "Click the Open help page button.",
                        "Download the file named ffmpeg-release-essentials.zip.",
                        "Unzip it and copy ffmpeg.exe into:",
                        f"{target_dir}",
                        "Press R to try again.",
                    ],
                    help_url="https://www.gyan.dev/ffmpeg/builds/",
                )
            if system == "darwin":
                raise StepFailure(
                    "FFmpeg install failed",
                    "FFmpeg did not install correctly.",
                    [
                        "Open the Terminal app.",
                        "Copy and paste this line: brew install ffmpeg",
                        "Press Enter, then press R to try again.",
                    ],
                    help_url="https://brew.sh",
                )
            raise StepFailure(
                "FFmpeg missing",
                "FFmpeg did not install correctly.",
                [
                    "Click the Open help page button.",
                    "Follow the install steps for your system.",
                    "Then press R to try again.",
                ],
                help_url="https://ffmpeg.org/download.html",
            )
        self._set_step_progress(100, 100, "FFmpeg ready")

    def _find_ffmpeg(self) -> Optional[str]:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            return ffmpeg_path
        if platform.system().lower().startswith("win"):
            local = self.repo_root / "synthesia2midi" / "ffmpeg" / "ffmpeg.exe"
            if local.exists():
                return str(local)
        return None

    async def _install_ffmpeg_windows(self) -> None:
        url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        target_dir = self.repo_root / "synthesia2midi" / "ffmpeg"
        target_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "ffmpeg.zip"
            self._log("Downloading FFmpeg...")
            await self._run_blocking(self._download_with_progress, url, zip_path, "Downloading FFmpeg")
            self._log("Extracting FFmpeg...")
            self._set_step_progress(100, 80, "Extracting FFmpeg")
            await self._run_blocking(self._extract_zip, zip_path, tmpdir)
            ffmpeg_exe = None
            for path in Path(tmpdir).rglob("ffmpeg.exe"):
                ffmpeg_exe = path
                break
            if not ffmpeg_exe:
                raise StepFailure(
                    "FFmpeg download failed",
                    "We downloaded FFmpeg but could not find the executable.",
                    [
                        "Click the Open help page button.",
                        "Download the file named ffmpeg-release-essentials.zip.",
                        "Unzip it and copy ffmpeg.exe into:",
                        f"{target_dir}",
                        "Press R to try again.",
                    ],
                    help_url="https://www.gyan.dev/ffmpeg/builds/",
                )
            self._set_step_progress(100, 90, "Installing FFmpeg")
            await self._run_blocking(shutil.copyfile, ffmpeg_exe, target_dir / "ffmpeg.exe")
            self._log(f"FFmpeg installed to {target_dir}")
            self._set_step_progress(100, 100, "FFmpeg ready")

    def _extract_zip(self, zip_path: Path, dest: str) -> None:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dest)

    def _download_with_progress(self, url: str, dest: Path, label: str) -> None:
        self.call_from_thread(self._set_step_progress, 100, 0, label)

        def reporthook(blocks: int, block_size: int, total_size: int) -> None:
            if total_size <= 0:
                return
            progress = min(75.0, (blocks * block_size) * 75.0 / total_size)
            self.call_from_thread(self._update_step_progress, progress)

        urllib.request.urlretrieve(url, dest, reporthook=reporthook)

    async def _install_ffmpeg_macos(self) -> None:
        brew = shutil.which("brew")
        if not brew:
            raise StepFailure(
                "Homebrew is not installed",
                "We need Homebrew to install FFmpeg on macOS.",
                [
                    "Click the Open help page button.",
                    "Follow the install steps on that page.",
                    "Then press R to try again.",
                ],
                help_url="https://brew.sh",
                fix_options=[FixOption("Install Homebrew for me", self._fix_install_homebrew)],
            )
        self._set_step_progress(100, 10, "Installing FFmpeg with Homebrew")
        code = await self._run_command([brew, "install", "ffmpeg"], cwd=self.repo_root)
        if code != 0:
            raise StepFailure(
                "FFmpeg install failed",
                "Homebrew could not install FFmpeg.",
                [
                    "Open the Terminal app.",
                    "Copy and paste this line: brew install ffmpeg",
                    "Press Enter, then press R to try again.",
                ],
                help_url="https://brew.sh",
            )
        self._set_step_progress(100, 100, "FFmpeg ready")

    async def step_rust(self) -> None:
        if self.test_mode:
            await self._simulate_progress("Installing Rust", steps=5)
            if self._test_should_fail("rust"):
                raise StepFailure(
                    "Rust install failed",
                    "We could not install Rust automatically.",
                    [
                        "Click the Open help page button.",
                        "Download and run the installer.",
                        "Then press R to try again.",
                    ],
                    help_url="https://www.rust-lang.org/tools/install",
                )
            self._set_step_progress(100, 100, "Rust ready (test mode)")
            self._log("Test mode: skipped Rust install.")
            return
        self._set_step_progress(100, 0, "Checking Rust")
        if self._find_cargo() and not self.force_reinstall:
            self._log("Rust already installed.")
            self._set_step_progress(100, 100, "Rust already installed")
            return
        if platform.system().lower().startswith("win"):
            await self._install_rust_windows()
        else:
            await self._install_rust_unix()
        if not self._find_cargo():
            raise StepFailure(
                "Rust install failed",
                "We could not find Rust after installation.",
                [
                    "Click the Open help page button.",
                    "Download and run the installer.",
                    "Then press R to try again.",
                ],
                help_url="https://www.rust-lang.org/tools/install",
            )
        self._set_step_progress(100, 100, "Rust ready")

    def _find_cargo(self) -> Optional[str]:
        cargo = shutil.which("cargo")
        if cargo:
            return cargo
        home = Path.home()
        if platform.system().lower().startswith("win"):
            candidate = home / ".cargo" / "bin" / "cargo.exe"
        else:
            candidate = home / ".cargo" / "bin" / "cargo"
        if candidate.exists():
            os.environ["PATH"] = f"{candidate.parent}{os.pathsep}{os.environ.get('PATH', '')}"
            return str(candidate)
        return None

    async def _install_rust_windows(self) -> None:
        winget = shutil.which("winget")
        if winget:
            self._set_step_progress(100, 10, "Installing Rust with winget")
            code = await self._run_command(
                [
                    winget,
                    "install",
                    "--id",
                    "Rustlang.Rustup",
                    "-e",
                    "--scope",
                    "user",
                    "--accept-source-agreements",
                    "--accept-package-agreements",
                ],
                cwd=self.repo_root,
            )
            if code == 0 and self._find_cargo():
                self._set_step_progress(100, 100, "Rust ready")
                return

        url = self._rustup_url_windows()
        with tempfile.TemporaryDirectory() as tmpdir:
            exe_path = Path(tmpdir) / "rustup-init.exe"
            self._log("Downloading Rust installer...")
            await self._run_blocking(self._download_with_progress, url, exe_path, "Downloading Rust installer")
            self._log("Running Rust installer...")
            self._set_step_progress(100, 90, "Installing Rust")
            code = await self._run_command([str(exe_path), "-y", "--profile", "minimal"], cwd=self.repo_root)
            if code != 0:
                raise StepFailure(
                    "Rust install failed",
                    "We could not install Rust automatically.",
                    [
                        "Click the Open help page button.",
                        "Download and run the installer.",
                        "Then press R to try again.",
                    ],
                    help_url="https://www.rust-lang.org/tools/install",
                )
            self._set_step_progress(100, 100, "Rust ready")

    def _rustup_url_windows(self) -> str:
        arch = platform.machine().lower()
        if "arm" in arch or "aarch64" in arch:
            return "https://static.rust-lang.org/rustup/dist/aarch64-pc-windows-msvc/rustup-init.exe"
        if arch in {"x86", "i386", "i686"}:
            return "https://static.rust-lang.org/rustup/dist/i686-pc-windows-msvc/rustup-init.exe"
        return "https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe"

    async def _install_rust_unix(self) -> None:
        command = "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"
        self._set_step_progress(100, 10, "Installing Rust")
        code = await self._run_command(["/bin/bash", "-c", command], cwd=self.repo_root)
        if code != 0:
            raise StepFailure(
                "Rust install failed",
                "We could not install Rust automatically.",
                [
                    "Click the Open help page button.",
                    "Follow the install steps on that page.",
                    "Then press R to try again.",
                ],
                help_url="https://www.rust-lang.org/tools/install",
            )
        self._find_cargo()
        self._set_step_progress(100, 100, "Rust ready")

    async def step_build_rust(self) -> None:
        if self.test_mode:
            await self._simulate_progress("Building Rust editor", steps=4)
            if self._test_should_fail("build"):
                rust_dir = self.repo_root / "tools" / "midi_touchup_editor_rust"
                raise StepFailure(
                    "Rust build failed",
                    "We could not build the MIDI Touch-Up Editor.",
                    [
                        "Press I to reinstall everything.",
                        "Option 1: Open the Terminal app.",
                        f"Option 1: Copy and paste this line: cd {rust_dir}",
                        "Option 1: Then run: cargo build --release",
                        "Option 1: After it finishes, press R to retry.",
                        "Option 2: Press D to show details and open logs/installer_tui.log for help.",
                    ],
                )
            self._set_step_progress(100, 100, "Rust editor built (test mode)")
            self._log("Test mode: skipped Rust build.")
            return
        self._set_step_progress(100, 0, "Building Rust editor")
        rust_dir = self.repo_root / "tools" / "midi_touchup_editor_rust"
        if not rust_dir.exists():
            self._log("Rust editor not present. Skipping build.")
            self._set_step_progress(100, 100, "Skipped build")
            return
        cargo = self._find_cargo()
        if not cargo:
            raise StepFailure(
                "Rust is missing",
                "We need Rust to build the MIDI Touch-Up Editor.",
                [
                    "Press R after Rust is installed.",
                ],
            )
        self._set_step_progress(100, 10, "Compiling Rust editor")
        code = await self._run_command([cargo, "build", "--release"], cwd=rust_dir)
        if code != 0:
            raise StepFailure(
                "Rust build failed",
                "We could not build the MIDI Touch-Up Editor.",
                [
                    "Press I to reinstall everything.",
                    "Option 1: Open the Terminal app.",
                    f"Option 1: Copy and paste this line: cd {rust_dir}",
                    "Option 1: Then run: cargo build --release",
                    "Option 1: After it finishes, press R to retry.",
                    "Option 2: Press D to show details and open logs/installer_tui.log for help.",
                ],
            )
        self._set_step_progress(100, 100, "Rust editor built")

    async def step_done(self) -> None:
        self._set_step_progress(100, 100, "All done")
        self._set_main(
            "Setup complete",
            "Everything is installed and ready to use.",
            "You can now close this window and start the app.",
        )
        self._log("Setup completed successfully.")


def main() -> None:
    InstallerApp().run()


if __name__ == "__main__":
    main()
