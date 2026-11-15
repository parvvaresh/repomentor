import os
from urllib.parse import quote

import requests
from dotenv import load_dotenv


# Read variables from the .env file
load_dotenv()

gitlab_url = (os.getenv("GITLAB_BASE_URL") or "").rstrip("/")
gitlab_token = os.getenv("GITLAB_TOKEN")
project_path = os.getenv("GITLAB_PROJECT_PATH")

# Stop early if any configuration is missing
if not all([gitlab_url, gitlab_token, project_path]):
    raise RuntimeError("One or more GitLab environment variables are missing.")

# GitLab requires the slash in "amir-hri/demo-company-app"
# to be URL-encoded as "%2F"
encoded_project_path = quote(project_path, safe="")

api_url = f"{gitlab_url}/api/v4/projects/{encoded_project_path}"

response = requests.get(
    api_url,
    headers={"PRIVATE-TOKEN": gitlab_token},
    timeout=15,
)

# Raise an error for responses such as 401 or 404
response.raise_for_status()

project = response.json()

print("Connection successful!")
print(f"Project ID: {project['id']}")
print(f"Project: {project['path_with_namespace']}")
print(f"Default branch: {project['default_branch']}")
print(f"Web URL: {project['web_url']}")