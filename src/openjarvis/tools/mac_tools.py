"""Mac-control tools from the voice assistant, available to every OpenJarvis agent.

Wraps the functions in ``openjarvis.assistant.mac_tools`` so ``jarvis chat`` and
``jarvis ask`` can tell the time anywhere, open apps and sites, set the volume,
control music, report system status and read the text you have selected.
"""

from __future__ import annotations

import sys
from typing import Any

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

# Screenshots and memory stay assistant-only: OpenJarvis tool results are text, and
# OpenJarvis has its own memory and shell tools.
MAC_TOOLS = (
    "get_time",
    "open_app",
    "open_url",
    "set_volume",
    "media_control",
    "system_status",
    "get_selected_text",
)


def _make(name: str) -> type[BaseTool]:
    class MacTool(BaseTool):
        tool_id = name

        def _tool(self) -> Any:
            from openjarvis.assistant import mac_tools

            return mac_tools.BY_NAME[name]

        @property
        def spec(self) -> ToolSpec:
            tool = self._tool()
            return ToolSpec(
                name=name,
                description=tool.description,
                parameters=tool.schema,
                category="mac",
            )

        def execute(self, **params: Any) -> ToolResult:
            output, is_error = self._tool().execute_simple(params)
            return ToolResult(tool_name=name, content=output, success=not is_error)

    MacTool.__name__ = MacTool.__qualname__ = f"Mac_{name}"
    return MacTool


if sys.platform == "darwin":
    for _name in MAC_TOOLS:
        ToolRegistry.register(_name)(_make(_name))

__all__ = ["MAC_TOOLS"]
