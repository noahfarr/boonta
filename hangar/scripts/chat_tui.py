import time
from queue import Empty, SimpleQueue

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

BG = "#16191e"
TEXT = "#e9ebee"
MUTED = "#9aa1aa"
DIM = "#6b7280"
HAIRLINE = "#292e36"
ACCENT = "#5b8def"
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
QUEUE_VISIBLE = 5


class ChatApp(App):
    CSS = f"""
    Screen {{ background: {BG}; color: {TEXT}; }}
    #log {{ height: 1fr; padding: 1 2; }}
    #welcome, #loading {{ width: 1fr; text-align: center; color: {DIM}; margin-top: 4; }}
    .user {{ color: {TEXT}; margin-top: 1; text-align: right; }}
    .assistant {{ color: {TEXT}; margin-top: 1; }}
    #queue {{ height: auto; max-height: 7; padding: 0 2; border-top: solid {HAIRLINE}; display: none; }}
    #queue Static {{ height: 1; }}
    #menu {{ height: auto; max-height: 8; padding: 0 2; border-top: solid {HAIRLINE}; display: none; }}
    #menu Static {{ height: 1; }}
    .menu-selected {{ background: {HAIRLINE}; }}
    #picker-hint {{ color: {DIM}; padding: 1 2 0 2; }}
    OptionList {{ background: {BG}; border: none; height: auto; max-height: 12; padding: 0; margin: 0 1; }}
    OptionList:focus {{ border: none; }}
    OptionList > .option-list--option {{ padding: 0 1; }}
    OptionList > .option-list--option-highlighted {{ background: {HAIRLINE}; }}
    Input {{ border: solid {HAIRLINE}; background: {BG}; }}
    Input:focus {{ border: solid $accent; }}
    #status {{ height: 1; padding: 0 3; margin-bottom: 1; }}
    #status-left {{ width: 1fr; }}
    #status-right {{ width: auto; }}
    """
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("escape", "cancel", "Cancel"),
        ("up", "history_previous", "Previous"),
        ("ctrl+p", "history_previous", "Previous"),
        ("down", "history_next", "Next"),
        ("ctrl+n", "history_next", "Next"),
    ]

    def __init__(
        self,
        engine_factory,
        title="chat",
        assistant="bot",
        model="",
        accent=ACCENT,
        commands=None,
        models=None,
    ):
        self._accent = accent
        super().__init__()
        self._engine_factory = engine_factory
        self._model = model
        self._previous_model = model
        self._commands = commands or {}
        self._models = models or []
        self.engine = None
        self._reply = None
        self._text = ""
        self._frame = 0
        self._cancel = False
        self._history = []
        self._history_index = None
        self._queued = []
        self._menu = []
        self._menu_index = 0
        self._tokens = 0
        self._generation_start = None
        self._queue = SimpleQueue()

    def get_css_variables(self):
        variables = super().get_css_variables()
        variables["accent"] = self._accent
        return variables

    def compose(self) -> ComposeResult:
        log = VerticalScroll(id="log")
        log.can_focus = False
        yield log
        yield Vertical(id="queue")
        yield Vertical(id="menu")
        yield Input(placeholder="Message", disabled=False)
        yield Horizontal(
            Static(self.status_left(), id="status-left"),
            Static(self.tps_text(None), id="status-right"),
            id="status",
        )

    def on_mount(self):
        self.mount_widget(Static("", id="loading"))
        self.query_one(Input).focus()
        self.set_interval(0.08, self.think)
        self.set_interval(0.08, self.loading)
        self.set_interval(1 / 30, self.pump)
        self.set_interval(1.0, self.refresh_status)
        self.load()

    def loading(self):
        if self.engine is not None:
            return
        widgets = list(self.query("#loading"))
        if not widgets:
            return
        self._frame = (self._frame + 1) % len(SPINNER)
        indicator = Text()
        indicator.append(SPINNER[self._frame] + " ", style=self._accent)
        indicator.append(f"loading {self._model.split('/')[-1]}", style=MUTED)
        widgets[0].update(indicator)

    def think(self):
        if self._reply is None:
            return
        thinking, answer = self.split(self._text)
        if thinking or answer:
            return
        self._frame = (self._frame + 1) % len(SPINNER)
        indicator = Text()
        indicator.append(SPINNER[self._frame] + " ", style=self._accent)
        indicator.append("thinking", style=MUTED)
        self._reply.update(indicator)

    @work(thread=True)
    def load(self):
        try:
            engine = self._engine_factory(self._model)
        except Exception as error:
            self.call_from_thread(self.load_failed, str(error))
            return
        self.call_from_thread(self.engine_ready, engine)

    def engine_ready(self, engine):
        self.engine = engine
        self._reply = None
        self.view().remove_children()
        name = getattr(engine, "repo_id", self._model).split("/")[-1]
        field = self.query_one(Input)
        field.placeholder = f"Message {name}"
        field.disabled = False
        field.focus()
        self.refresh_status()
        self.welcome()
        self.dequeue()

    def load_failed(self, error):
        attempted = self._model.split("/")[-1]
        reason = error.splitlines()[0][:80] if error else "failed"
        self.engine = None
        if self._previous_model and self._previous_model != self._model:
            self.begin_load(self._previous_model)
            self.notice(f"could not load {attempted}: {reason}")
        else:
            self.view().remove_children()
            self.query_one(Input).placeholder = "no model — /model to choose"
            self.notice(f"could not load {attempted}: {reason}")

    def welcome(self):
        self.mount_widget(Static("How can I help?", id="welcome"))

    def action_cancel(self):
        if self._menu:
            self._menu = []
            self.render_menu()
            return
        if self.close_picker():
            return
        if self._reply is not None:
            self._cancel = True

    def action_history_previous(self):
        if self._menu:
            self._menu_index = (self._menu_index - 1) % len(self._menu)
            self.render_menu()
            return
        if not self._history:
            return
        if self._history_index is None:
            self._history_index = len(self._history)
        self._history_index = max(0, self._history_index - 1)
        self.set_input(self._history[self._history_index])

    def action_history_next(self):
        if self._menu:
            self._menu_index = (self._menu_index + 1) % len(self._menu)
            self.render_menu()
            return
        if self._history_index is None:
            return
        self._history_index += 1
        if self._history_index >= len(self._history):
            self._history_index = None
            self.set_input("")
        else:
            self.set_input(self._history[self._history_index])

    def set_input(self, value):
        field = self.query_one(Input)
        field.value = value
        field.cursor_position = len(value)

    @staticmethod
    def split(text):
        if "<think>" not in text:
            return "", text
        after = text.split("<think>", 1)[1]
        if "</think>" in after:
            thinking, answer = after.split("</think>", 1)
            return thinking.strip(), answer.strip()
        return after.strip(), ""

    def render_reply(self):
        thinking, answer = self.split(self._text)
        result = Text()
        if thinking:
            result.append(thinking, style=DIM)
            if answer:
                result.append("\n\n")
        if answer:
            result.append("◆ ", style=self._accent)
            result.append(answer, style=TEXT)
        self._reply.update(result)

    def view(self):
        return self.query_one("#log", VerticalScroll)

    def mount_widget(self, widget):
        self.view().mount(widget)
        self.view().scroll_end(animate=False)

    def on_input_submitted(self, event: Input.Submitted):
        if self._menu:
            command = self._menu[self._menu_index]
            event.input.value = ""
            self._menu = []
            self.render_menu()
            self.submit(command)
            return
        message = event.value.strip()
        if not message:
            return
        event.input.value = ""
        self.submit(message)

    def submit(self, text):
        if text in ("/exit", "/quit"):
            self.exit()
            return
        if self.engine is None or self._reply is not None:
            self.enqueue(text)
            return
        if text.startswith("/"):
            self.command(text)
        else:
            self.send(text)

    def command(self, message):
        parts = message.split(maxsplit=1)
        name = parts[0]
        argument = parts[1].strip() if len(parts) > 1 else ""
        if name in ("/exit", "/quit"):
            self.exit()
            return
        if self.engine is None:
            self.notice("still loading")
            return
        if name == "/help":
            self.notice(self.help_text())
        elif name in ("/clear", "/reset"):
            self.clear()
        elif name == "/model":
            if argument:
                self.switch_model(argument)
            else:
                self.show_models()
        elif name in self._commands:
            result = self._commands[name][1](self, argument)
            if result:
                self.notice(result)
        else:
            self.notice(f"unknown command: {name} (try /help)")

    def show_models(self):
        if not self._models:
            self.notice(f"model: {self._model}")
            return
        if list(self.query("#picker")):
            return
        self.query("#welcome").remove()
        self.mount_widget(
            Static(
                Text("select a model  ·  Enter to switch, Esc to cancel", style=DIM),
                id="picker-hint",
            )
        )
        options = []
        for model in self._models:
            label = Text()
            current = model == self._model
            label.append(
                "● " if current else "○ ", style=self._accent if current else DIM
            )
            label.append(model, style=TEXT if current else MUTED)
            options.append(Option(label, id=model))
        picker = OptionList(*options, id="picker")
        self.mount_widget(picker)
        if self._model in self._models:
            picker.highlighted = self._models.index(self._model)
        picker.focus()

    def close_picker(self):
        widgets = list(self.query("#picker")) + list(self.query("#picker-hint"))
        if not widgets:
            return False
        for widget in widgets:
            widget.remove()
        self.query_one(Input).focus()
        return True

    def on_option_list_option_selected(self, event: OptionList.OptionSelected):
        model = event.option.id
        self.close_picker()
        if model and model != self._model:
            self.switch_model(model)

    def command_rows(self):
        rows = {
            "/help": "show commands",
            "/model": "pick or switch a model",
            "/clear": "clear the conversation",
            "/exit": "quit",
        }
        for name, (description, _) in self._commands.items():
            rows[name] = description
        return rows

    def help_text(self):
        rows = self.command_rows()
        width = max(len(name) for name in rows)
        return "\n".join(
            f"{name.ljust(width)}  {description}" for name, description in rows.items()
        )

    def on_input_changed(self, event: Input.Changed):
        value = event.value
        if value.startswith("/") and " " not in value:
            self._menu = [
                name for name in self.command_rows() if name.startswith(value)
            ]
        else:
            self._menu = []
        self._menu_index = 0
        self.render_menu()

    def render_menu(self):
        container = self.query_one("#menu", Vertical)
        container.remove_children()
        rows = self.command_rows()
        for index, name in enumerate(self._menu):
            line = Text()
            selected = index == self._menu_index
            line.append(name, style=self._accent if selected else TEXT)
            line.append("  ")
            line.append(rows[name], style=DIM)
            row = Static(line)
            if selected:
                row.add_class("menu-selected")
            container.mount(row)
        container.display = bool(self._menu)

    def notice(self, text):
        self.query("#welcome").remove()
        self.mount_widget(Static(Text(text, style=DIM), classes="assistant"))

    def clear(self):
        self.engine.reset()
        self._queued = []
        self.refresh_queue()
        self.view().remove_children()
        self.welcome()

    def switch_model(self, model):
        self._previous_model = self._model
        if hasattr(self.engine, "unload"):
            self.engine.unload()
        self._queued = []
        self.refresh_queue()
        self.begin_load(model)

    def begin_load(self, model):
        self.engine = None
        self._model = model
        self._reply = None
        self.query_one(Input).placeholder = f"loading {model.split('/')[-1]}..."
        self.view().remove_children()
        self.mount_widget(Static("", id="loading"))
        self.load()

    def prompt_text(self, message):
        text = Text()
        text.append("❯ ", style=f"bold {self._accent}")
        text.append(message)
        return text

    def pending_text(self, message):
        text = Text()
        text.append("❯ ", style=DIM)
        text.append(message, style=DIM)
        return text

    def refresh_queue(self):
        container = self.query_one("#queue", Vertical)
        container.remove_children()
        shown = self._queued[:QUEUE_VISIBLE]
        for message in shown:
            container.mount(Static(self.pending_text(message)))
        extra = len(self._queued) - len(shown)
        if extra:
            container.mount(Static(Text(f"+{extra} more", style=DIM)))
        container.display = bool(self._queued)

    def enqueue(self, message):
        self._queued.append(message)
        self.refresh_queue()

    def dequeue(self):
        while self._queued and self._reply is None:
            item = self._queued.pop(0)
            self.refresh_queue()
            if item.startswith("/"):
                self.command(item)
            else:
                self.send(item)

    def send(self, message):
        self.query("#welcome").remove()
        self._history.append(message)
        self._history_index = None
        self._cancel = False
        self.mount_widget(Static(self.prompt_text(message), classes="user"))
        self._text = ""
        self._tokens = 0
        self._generation_start = None
        self._reply = Static("", classes="assistant")
        self.mount_widget(self._reply)
        self.generate(message)

    @work(thread=True)
    def generate(self, message):
        try:
            for delta in self.engine.stream(message):
                if self._cancel:
                    break
                self._queue.put(delta)
        except ValueError as error:
            self._queue.put(f"\n[{error}]")
        self._queue.put(None)

    def pump(self):
        if self._reply is None:
            return
        chunks, done = [], False
        try:
            while True:
                item = self._queue.get_nowait()
                if item is None:
                    done = True
                    break
                chunks.append(item)
        except Empty:
            pass
        if chunks:
            if self._generation_start is None:
                self._generation_start = time.monotonic()
            self._tokens += len(chunks)
            self._text += "".join(chunks)
            elapsed = time.monotonic() - self._generation_start
            if self._tokens > 2 and elapsed > 0.1:
                self.set_tps(self._tokens / elapsed)
            thinking, answer = self.split(self._text)
            if thinking or answer:
                self.render_reply()
                self.view().scroll_end(animate=False)
        if done:
            thinking, answer = self.split(self._text)
            if not thinking and not answer and self._reply is not None:
                self._reply.remove()
            self._reply = None
            self._cancel = False
            tps = getattr(self.engine, "tps", 0.0)
            if tps:
                self.set_tps(tps)
            self.query_one(Input).focus()
            self.dequeue()

    def tps_text(self, tps):
        text = Text()
        if tps is None:
            text.append("-", style=DIM)
        else:
            text.append(f"{tps:.0f}", style=self._accent)
        text.append(" tokens/s", style=DIM)
        return text

    def set_tps(self, tps):
        self.query_one("#status-right", Static).update(self.tps_text(tps))

    def status_left(self):
        text = Text()
        text.append(self._model.split("/")[-1], style=MUTED)
        engine = self.engine
        if engine is None:
            return text
        think = getattr(engine, "think", None)
        if think is not None:
            text.append("  ·  ", style=DIM)
            text.append(f"thinking {'on' if think else 'off'}", style=DIM)
        used = getattr(engine, "used", None)
        limit = getattr(engine, "context_length", None)
        if used is not None and limit:
            text.append("  ·  ", style=DIM)
            text.append(f"ctx {used / 1000:.1f}k/{limit / 1000:.0f}k", style=DIM)
        vram = getattr(engine, "vram", None)
        if callable(vram):
            try:
                text.append("  ·  ", style=DIM)
                text.append(f"vram {vram()}", style=DIM)
            except Exception:
                pass
        return text

    def refresh_status(self):
        self.query_one("#status-left", Static).update(self.status_left())
