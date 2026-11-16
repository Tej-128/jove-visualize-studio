# JoVE Visualize Studio

Public source for an invited-user pilot: script matching, timed narration, staged visual footage selection, and Premiere-free video assembly. The public repository contains no library binaries, scripts, generated outputs, or credentials.

Deploy this separate application from branch `main`, main file `studio/app.py`, Python 3.11. Configure secure settings using `studio/.streamlit/secrets.example.toml`. Never commit real keys.

The source passes 21 local checks. A first cloud trial has passed matching and narration and reached visual review; a complete scientific edit is still pending. See [setup, workflow, and limitations](studio/README.md).

Existing JoVE audio and Visualizer tools remain separate. Credentials, caption indexes, source videos, and generated reports are excluded from this repository. The hosted application must pass live sign-in and access-denial checks before company media is processed. Public source is not a security certification.

