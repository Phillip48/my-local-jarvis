"""The two interchangeable brains: Claude (cloud) and Ollama (local).

Both run the same loop: stream text back as it's generated, run any tools the model asks
for, feed the results back, and repeat until the model gives a final answer.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any, cast

import anthropic
import ollama
from anthropic.types.beta import (
    BetaMessageParam,
    BetaOutputConfigParam,
    BetaToolUnionParam,
)

from . import config, memory
from . import mac_tools as tools

PERSONA = """You are J.A.R.V.I.S., a personal AI assistant running on {user}'s Mac. You are \
calm, capable and dryly witty, in the manner of a refined British butler. Address the user as \
"{user}".

Your replies are spoken aloud by a text-to-speech voice, so:
- Keep answers short: usually one to three sentences. Offer detail only if asked.
- Write plain spoken English. No markdown, bullet points, headings, code blocks, emoji or URLs.
- Say numbers and units the way a person would say them.

Use your tools to act on the user's behalf rather than explaining how they could do it \
themselves. After using a tool, briefly confirm what you did. When the user tells you something \
about themselves worth remembering (preferences, people, projects, routines), save it with the \
remember tool without making a fuss about it.

Search the web before answering any factual question unless you are certain your answer is \
correct and still current. Always search for these, even if you think you know the answer, \
because they change: who currently holds a job or title (CEOs, leaders, officials), recent \
winners and results, prices, statistics and populations, release dates and latest versions, \
schedules and opening hours, news, and anything that may have changed since your training. \
Never guess and never answer such questions from memory, and don't ask before searching. \
Don't search for timeless common knowledge, arithmetic, small talk, or controlling the Mac.

Latency-sensitive; begin your visible answer immediately."""


def system_prompt() -> str:
    return (
        PERSONA.format(user=config.USER_NAME)
        + f"\n\nToday's date is {date.today():%A %d %B %Y}."
        + f"\n\nWhat you remember about the user:\n{memory.as_prompt()}"
    )


@dataclass
class Events:
    on_text: Callable[[str], None]  # streamed reply text
    on_tool: Callable[[str, dict], None]  # a tool is about to run
    confirm: Callable[[str, str], bool]  # ask the user to approve a tool call
    _last: str = ""  # last character of reply text sent so far
    _new_part: bool = False

    def text(self, delta: str) -> None:
        """Send reply text, keeping a space between pieces generated separately (e.g. before
        and after a tool call), which models start without one."""
        if self._new_part and self._last and not self._last.isspace() and not delta[:1].isspace():
            delta = " " + delta
        self._new_part = False
        self._last = delta[-1:] or self._last
        self.on_text(delta)

    def new_part(self) -> None:
        self._new_part = True


class ClaudeBrain:
    name = "claude"

    def __init__(self) -> None:
        self.client = anthropic.Anthropic()
        self.tools: list[BetaToolUnionParam] = [
            {"name": t.name, "description": t.description, "input_schema": t.schema}
            for t in tools.TOOLS
        ]
        self.tools.append({"type": "web_search_20260209", "name": "web_search", "max_uses": 3})
        self.output_config = cast(BetaOutputConfigParam, {"effort": config.CLAUDE_EFFORT})
        self.reset()

    @property
    def label(self) -> str:
        return config.CLAUDE_MODEL

    def warm(self) -> None:
        pass  # nothing to load

    def reset(self) -> None:
        self.system = system_prompt()
        self.messages: list[BetaMessageParam] = []

    def respond(self, text: str, ev: Events) -> None:
        start = len(self.messages)
        self.messages.append({"role": "user", "content": text})
        try:
            self._loop(ev, start)
        except BaseException:
            del self.messages[start:]  # drop the failed turn so history stays valid
            raise

    def _loop(self, ev: Events, start: int) -> None:
        for _ in range(12):
            with self.client.beta.messages.stream(
                model=config.CLAUDE_MODEL,
                max_tokens=64000,
                system=self.system,
                tools=self.tools,
                messages=self.messages,
                output_config=self.output_config,
                cache_control={"type": "ephemeral"},
                # On a safety decline, retry server-side on Anthropic's recommended fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                for event in stream:
                    if event.type == "content_block_start" and event.content_block.type == "text":
                        ev.new_part()
                    elif event.type == "text":
                        ev.text(event.text)
                msg = stream.get_final_message()

            if msg.stop_reason == "refusal":
                del self.messages[start:]
                ev.new_part()
                ev.text("I'm afraid I can't help with that one.")
                return

            self.messages.append({"role": "assistant", "content": msg.content})
            if msg.stop_reason == "pause_turn":  # long web search; let the server continue
                continue

            tool_uses = [b for b in msg.content if b.type == "tool_use"]
            if msg.stop_reason != "tool_use" or not tool_uses:
                return

            results = []
            for block in tool_uses:
                ev.on_tool(block.name, block.input)
                output, is_error = tools.execute(block.name, block.input, ev.confirm)
                if isinstance(output, tools.Image):
                    content: Any = [
                        {"type": "text", "text": output.text},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": output.b64},
                        },
                    ]
                else:
                    content = output
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                    "is_error": is_error,
                })
            self.messages.append({"role": "user", "content": results})
        ev.new_part()
        ev.text("I seem to be going in circles, so I'll stop there.")


class OllamaBrain:
    name = "ollama"

    def __init__(self) -> None:
        self.client = ollama.Client()
        self.tools = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.schema},
            }
            for t in tools.TOOLS + tools.WEB_TOOLS
        ]
        self.reset()

    @property
    def label(self) -> str:
        return config.OLLAMA_MODEL

    def reset(self) -> None:
        self.messages: list = [{"role": "system", "content": system_prompt()}]

    def warm(self) -> None:
        """Load the model and have it read the instructions, tools and conversation so far,
        in the background, while the user is still talking. The reply then only has to
        read the new question: about 1 s instead of 7 (or 20 when the model wasn't loaded)."""
        messages = list(self.messages)

        def run() -> None:
            try:
                self.client.chat(
                    model=config.OLLAMA_MODEL,
                    messages=messages,
                    tools=self.tools,
                    think=False,
                    options={"num_ctx": config.OLLAMA_CONTEXT, "num_predict": 1},
                    keep_alive=config.OLLAMA_KEEP_ALIVE,
                )
            except Exception:  # noqa: BLE001 - only a head start; the real request reports errors
                pass

        threading.Thread(target=run, daemon=True).start()

    def respond(self, text: str, ev: Events) -> None:
        start = len(self.messages)
        self.messages.append({"role": "user", "content": text})
        try:
            self._loop(ev)
        except BaseException:
            del self.messages[start:]
            raise

    def _loop(self, ev: Events) -> None:
        for step in range(8):
            content, calls = [], []
            ev.new_part()
            for chunk in self.client.chat(
                model=config.OLLAMA_MODEL,
                messages=self.messages,
                tools=self.tools,
                stream=True,
                # Reasoning is slow, so it is off for plain chat and optional after tools.
                think=config.OLLAMA_THINK and step > 0,
                options={"num_ctx": config.OLLAMA_CONTEXT},  # room for web results
                keep_alive=config.OLLAMA_KEEP_ALIVE,
            ):
                if chunk.message.content:
                    content.append(chunk.message.content)
                    ev.text(chunk.message.content)
                if chunk.message.tool_calls:
                    calls.extend(chunk.message.tool_calls)

            self.messages.append(
                ollama.Message(role="assistant", content="".join(content), tool_calls=calls or None)
            )
            if not calls:
                return
            for call in calls:
                args = dict(call.function.arguments or {})
                ev.on_tool(call.function.name, args)
                output, _ = tools.execute(call.function.name, args, ev.confirm)
                msg = {"role": "tool", "content": output, "tool_name": call.function.name}
                if isinstance(output, tools.Image):
                    # Screenshots fill the small context fast, so keep only the newest.
                    for old in self.messages:
                        if isinstance(old, dict):
                            old.pop("images", None)
                    msg.update(content=output.text, images=[output.jpeg])
                self.messages.append(msg)
        ev.new_part()
        ev.text("I seem to be going in circles, so I'll stop there.")


BRAINS = {"claude": ClaudeBrain, "ollama": OllamaBrain}
