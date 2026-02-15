#!/usr/bin/env python3
"""
DadAI v2 — Deploy to Hugging Face Spaces

Creates (or updates) a Hugging Face Space with the DadAI Gradio app.

Usage:
    python scripts/deploy_to_hf.py --token YOUR_HF_TOKEN

The script will:
1. Create a Space repo on Hugging Face (if it doesn't exist)
2. Upload the app files from hf-space/
3. HF builds and deploys automatically
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder


def main():
    parser = argparse.ArgumentParser(description="Deploy DadAI to HF Spaces")
    parser.add_argument("--token", required=True,
                        help="Hugging Face write token")
    parser.add_argument("--repo-id", default=None,
                        help="HF repo ID (default: <username>/DadAI)")
    args = parser.parse_args()

    api = HfApi(token=args.token)

    # Get username
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = args.repo_id or f"{username}/DadAI"

    print(f"Deploying to: https://huggingface.co/spaces/{repo_id}")

    # Create the Space (no-op if it already exists)
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            token=args.token,
            exist_ok=True,
        )
        print(f"Space repo ready: {repo_id}")
    except Exception as e:
        print(f"Note: {e}")

    # Upload the hf-space directory
    space_dir = Path(__file__).parent.parent / "hf-space"
    print(f"Uploading files from {space_dir}...")

    upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(space_dir),
        token=args.token,
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\nDone! Your Space is deploying at:\n  {url}")
    print("\nIt may take a few minutes for the first build.")
    print("The model (~3.8 GB) needs to download on first launch.")


if __name__ == "__main__":
    main()
