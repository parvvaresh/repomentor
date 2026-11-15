import os
from pathlib import PurePosixPath
from urllib.parse import quote

import requests
from dotenv import load_dotenv


load_dotenv()

gitlab_url = (os.getenv("GITLAB_BASE_URL") or "").rstrip("/")
gitlab_token = os.getenv("GITLAB_TOKEN")
project_path = os.getenv("GITLAB_PROJECT_PATH")

if not all([gitlab_url, gitlab_token, project_path]):
    raise RuntimeError("One or more GitLab environment variables are missing.")

encoded_project_path = quote(project_path, safe="")
tree_url = (
    f"{gitlab_url}/api/v4/projects/"
    f"{encoded_project_path}/repository/tree"
)

headers = {"PRIVATE-TOKEN": gitlab_token}

allowed_extensions = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".md",
    ".yml",
    ".yaml",
    ".toml",
    ".sql",
}

excluded_directories = {
    "node_modules",
    ".venv",
    "dist",
    "build",
    "coverage",
    "vendor",
}


def is_useful_file(file_path: str) -> bool:
    """Return True when a repository file is useful for our knowledge base."""

    path = PurePosixPath(file_path)

    if any(directory in path.parts for directory in excluded_directories):
        return False

    if path.name == "Dockerfile":
        return True

    return path.suffix.lower() in allowed_extensions


all_items = []
page = 1

while True:
    response = requests.get(
        tree_url,
        headers=headers,
        params={
            "recursive": "true",
            "per_page": 100,
            "page": page,
        },
        timeout=30,
    )
    response.raise_for_status()

    all_items.extend(response.json())

    next_page = response.headers.get("X-Next-Page")

    if not next_page:
        break

    page = int(next_page)


repository_files = [
    item["path"]
    for item in all_items
    if item["type"] == "blob" and is_useful_file(item["path"])
]

print(f"GitLab returned {len(all_items)} files and directories.")
print(f"We selected {len(repository_files)} useful files.\n")

for file_path in repository_files[:30]:
    print(file_path)

if len(repository_files) > 30:
    print(f"\n...and {len(repository_files) - 30} more files.")