import base64
import os
from urllib.parse import quote

import requests
from dotenv import load_dotenv


load_dotenv()

gitlab_url = (os.getenv("GITLAB_BASE_URL") or "").rstrip("/")
gitlab_token = os.getenv("GITLAB_TOKEN")
project_path = os.getenv("GITLAB_PROJECT_PATH")

file_path = "README.md"

encoded_project = quote(project_path, safe="")
encoded_file = quote(file_path, safe="")

api_url = (
    f"{gitlab_url}/api/v4/projects/{encoded_project}"
    f"/repository/files/{encoded_file}"
)

response = requests.get(
    api_url,
    headers={"PRIVATE-TOKEN": gitlab_token},
    params={"ref": "HEAD"},
    timeout=30,
)
response.raise_for_status()

file_data = response.json()

# GitLab sends the file content using Base64 encoding
content = base64.b64decode(file_data["content"]).decode("utf-8")

print(f"File: {file_data['file_path']}")
print(f"Last commit: {file_data['last_commit_id']}")
print("\nFirst 500 characters:\n")
print(content[:500])