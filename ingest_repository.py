import base64
import json
import os
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import requests
from dotenv import load_dotenv


load_dotenv()

DEFAULT_OUTPUT_PATH = Path("data/repository_files.jsonl")

allowed_extensions = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".md", ".yml", ".yaml", ".toml", ".sql",
}

excluded_directories = {
    "node_modules", ".venv", "dist",
    "build", "coverage", "vendor",
}


def is_useful_file(file_path: str) -> bool:
    path = PurePosixPath(file_path)

    if any(part in excluded_directories for part in path.parts):
        return False

    return path.name == "Dockerfile" or path.suffix.lower() in allowed_extensions


def get_gitlab_settings() -> tuple[str, str]:
    gitlab_url = (os.getenv("GITLAB_BASE_URL") or "").rstrip("/")
    gitlab_token = os.getenv("GITLAB_TOKEN")

    if not gitlab_url or not gitlab_token:
        raise RuntimeError(
            "GITLAB_BASE_URL and GITLAB_TOKEN must be set in .env."
        )

    return gitlab_url, gitlab_token


def list_repository_files(
    session: requests.Session,
    gitlab_url: str,
    project_path: str,
) -> list[dict]:
    encoded_project = quote(project_path, safe="")
    url = (
        f"{gitlab_url}/api/v4/projects/"
        f"{encoded_project}/repository/tree"
    )

    files = []
    page = 1

    while True:
        response = session.get(
            url,
            params={
                "recursive": "true",
                "per_page": 100,
                "page": page,
            },
            timeout=30,
        )
        response.raise_for_status()

        for item in response.json():
            if item["type"] == "blob" and is_useful_file(item["path"]):
                files.append(item)

        next_page = response.headers.get("X-Next-Page")

        if not next_page:
            break

        page = int(next_page)

    return files


def download_file(
    session: requests.Session,
    gitlab_url: str,
    project_path: str,
    file_path: str,
) -> dict:
    encoded_project = quote(project_path, safe="")
    encoded_file = quote(file_path, safe="")

    url = (
        f"{gitlab_url}/api/v4/projects/{encoded_project}"
        f"/repository/files/{encoded_file}"
    )

    response = session.get(
        url,
        params={"ref": "HEAD"},
        timeout=30,
    )
    response.raise_for_status()

    file_data = response.json()

    content = base64.b64decode(
        file_data["content"]
    ).decode("utf-8", errors="replace")

    return {
        "project_path": project_path,
        "file_path": file_data["file_path"],
        "blob_id": file_data["blob_id"],
        "last_commit_id": file_data["last_commit_id"],
        "size": file_data["size"],
        "content": content,
    }


def ingest_repository(
    project_path: str,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    progress: Callable[[int, int, str, bool], None] | None = None,
) -> tuple[int, int]:
    """
    Download every useful file of a GitLab project into a JSON Lines
    corpus. Returns the number of downloaded and failed files.

    `progress(position, total, file_path, succeeded)` is called after
    each file so callers can report progress.
    """

    gitlab_url, gitlab_token = get_gitlab_settings()

    session = requests.Session()
    session.headers.update({"PRIVATE-TOKEN": gitlab_token})

    repository_files = list_repository_files(session, gitlab_url, project_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for position, item in enumerate(repository_files, start=1):
            file_path = item["path"]

            try:
                record = download_file(
                    session,
                    gitlab_url,
                    project_path,
                    file_path,
                )
                output_file.write(json.dumps(record) + "\n")
                successful += 1
                succeeded = True

            except requests.RequestException:
                failed += 1
                succeeded = False

            if progress:
                progress(position, len(repository_files), file_path, succeeded)

    return successful, failed


def main() -> None:
    project_path = os.getenv("GITLAB_PROJECT_PATH")

    if not project_path:
        raise RuntimeError("GITLAB_PROJECT_PATH is missing from .env.")

    def report(position: int, total: int, file_path: str, succeeded: bool):
        status = "Downloaded" if succeeded else "Failed"
        print(f"[{position}/{total}] {status} {file_path}")

    successful, failed = ingest_repository(
        project_path,
        DEFAULT_OUTPUT_PATH,
        report,
    )

    print("\nIngestion complete.")
    print(f"Downloaded: {successful}")
    print(f"Failed: {failed}")
    print(f"Saved to: {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
