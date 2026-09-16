#!/usr/bin/env python3
"""Download the Biohub competition data with the official Kaggle client."""

from __future__ import annotations

import argparse
from pathlib import Path

import kagglehub
from kagglehub.config import get_kaggle_credentials, set_kaggle_api_token


def bridge_cli_oauth() -> None:
    """Let kagglehub reuse an OAuth login created by the official Kaggle CLI.

    kagglehub 1.0 currently checks API-token/legacy locations but does not read
    ``~/.kaggle/credentials.json`` itself.  The Kaggle CLI does.  Keep the
    short-lived access token in memory instead of printing or duplicating it.
    """
    if get_kaggle_credentials() is not None:
        return
    try:
        from kagglesdk import KaggleClient, KaggleCredentials
    except ImportError:
        return
    client = KaggleClient()
    credentials = KaggleCredentials.load(client=client)
    if credentials is not None:
        token = credentials.get_access_token()
        if token:
            set_kaggle_api_token(token)


COMPETITION = "biohub-cell-tracking-during-development"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Destination managed by kagglehub (default: data/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a fresh download instead of using Kaggle's cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bridge_cli_oauth()
    path = kagglehub.competition_download(
        COMPETITION,
        output_dir=str(args.output_dir.resolve()),
        force_download=args.force,
    )
    print("Path to competition files:", path)


if __name__ == "__main__":
    main()
