"""``jarvis assistant`` — the always-listening voice assistant and arc-reactor HUD."""

from __future__ import annotations

import click


@click.command()
def assistant() -> None:
    """Start the voice assistant: "Hey Jarvis" wake word, HUD and Mac tools."""
    from openjarvis.assistant import main

    main()
