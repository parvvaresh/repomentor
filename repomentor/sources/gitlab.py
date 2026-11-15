"""GitLab REST API (gitlab.com or self-hosted)."""

import base64
from urllib.parse import quote

import requests

from .base import (
    AccessDenied,
    FileEntry,
    NotFound,
    RepositorySource,
    SourceError,
    UserInfo,
    is_useful_file,
)


TIMEOUT = 30


def check_response(response: requests.Response, what: str) -> None:
    if response.ok:
        return

    if response.status_code in (401, 403):
        raise AccessDenied(f"GitLab refused access to {what}.")

    if response.status_code == 404:
        raise NotFound(f"GitLab could not find {what}, or the token cannot see it.")

    raise SourceError(
        f"GitLab returned {response.status_code} for {what}: "
        f"{response.text[:200]}"
    )


class GitLabSource(RepositorySource):
    provider = "gitlab"

    def __init__(self, base_url: str, token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

        if token:
            self.session.headers["PRIVATE-TOKEN"] = token

    # --- Helpers ------------------------------------------------------------

    def _project_url(self, project: str) -> str:
        return f"{self.base_url}/api/v4/projects/{quote(project, safe='')}"

    def _get(self, url: str, what: str, **params) -> requests.Response:
        try:
            response = self.session.get(url, params=params, timeout=TIMEOUT)
        except requests.RequestException as error:
            raise SourceError(f"Could not reach GitLab: {error}") from error

        check_response(response, what)
        return response

    def _paginate(self, url: str, what: str, limit: int | None = None, **params):
        page = 1
        count = 0

        while True:
            response = self._get(url, what, per_page=100, page=page, **params)

            for item in response.json():
                yield item
                count += 1

                if limit is not None and count >= limit:
                    return

            next_page = response.headers.get("X-Next-Page")

            if not next_page:
                return

            page = int(next_page)

    # --- Files --------------------------------------------------------------

    def web_url(self, project: str) -> str:
        return f"{self.base_url}/{project}"

    def head_commit(self, project: str) -> str:
        details = self._get(self._project_url(project), project).json()
        branch = details.get("default_branch") or "HEAD"
        commit = self._get(
            f"{self._project_url(project)}/repository/commits/{quote(branch, safe='')}",
            f"the {branch} branch of {project}",
        ).json()
        return commit["id"]

    def list_files(self, project: str, ref: str) -> list[FileEntry]:
        return [
            FileEntry(path=item["path"], blob_id=item["id"])
            for item in self._paginate(
                f"{self._project_url(project)}/repository/tree",
                project,
                recursive="true",
                ref=ref,
            )
            if item["type"] == "blob" and is_useful_file(item["path"])
        ]

    def read_file(self, project: str, file_path: str, ref: str) -> dict:
        data = self._get(
            f"{self._project_url(project)}/repository/files/"
            f"{quote(file_path, safe='')}",
            file_path,
            ref=ref,
        ).json()

        return {
            "content": base64.b64decode(data["content"]).decode(
                "utf-8", errors="replace"
            ),
            "blob_id": data["blob_id"],
            "last_commit_id": data["last_commit_id"],
            "size": data["size"],
        }

    # --- History ------------------------------------------------------------

    def _commit(self, project: str, item: dict) -> dict:
        return {
            "sha": item["id"],
            "title": item.get("title") or "",
            "message": item.get("message") or "",
            "author_name": item.get("author_name") or "",
            "author_email": item.get("author_email") or "",
            "date": item.get("committed_date") or item.get("created_at") or "",
            "url": item.get("web_url")
            or f"{self.web_url(project)}/-/commit/{item['id']}",
        }

    def list_commits(self, project: str, limit: int) -> list[dict]:
        return [
            self._commit(project, item)
            for item in self._paginate(
                f"{self._project_url(project)}/repository/commits",
                f"commits of {project}",
                limit=limit,
            )
        ]

    def file_history(self, project: str, file_path: str, limit: int) -> list[dict]:
        return [
            self._commit(project, item)
            for item in self._paginate(
                f"{self._project_url(project)}/repository/commits",
                f"history of {file_path}",
                limit=limit,
                path=file_path,
            )
        ]

    def list_merge_requests(self, project: str, limit: int) -> list[dict]:
        return [
            {
                "number": item["iid"],
                "title": item.get("title") or "",
                "description": item.get("description") or "",
                "author": (item.get("author") or {}).get("name", ""),
                "merged_at": item.get("merged_at") or "",
                "url": item["web_url"],
            }
            for item in self._paginate(
                f"{self._project_url(project)}/merge_requests",
                f"merge requests of {project}",
                limit=limit,
                state="merged",
                order_by="updated_at",
            )
        ]

    # --- Reviews and changes -----------------------------------------------

    def get_merge_request(self, project: str, number: int) -> dict:
        url = f"{self._project_url(project)}/merge_requests/{number}"
        details = self._get(url, f"merge request !{number}").json()
        diffs = list(self._paginate(f"{url}/diffs", f"changes of !{number}"))

        return {
            "number": number,
            "title": details.get("title") or "",
            "description": details.get("description") or "",
            "author": (details.get("author") or {}).get("name", ""),
            "url": details["web_url"],
            "source_branch": details.get("source_branch"),
            "target_branch": details.get("target_branch"),
            "changes": [
                {
                    "path": change["new_path"],
                    "old_path": change["old_path"],
                    "diff": change.get("diff") or "",
                    "new_file": change.get("new_file", False),
                    "deleted_file": change.get("deleted_file", False),
                }
                for change in diffs
            ],
        }

    def create_merge_request(self, project, branch, title, description, files):
        details = self._get(self._project_url(project), project).json()
        target = details.get("default_branch") or "main"

        actions = []

        for file_path, content in files.items():
            exists = self.session.head(
                f"{self._project_url(project)}/repository/files/"
                f"{quote(file_path, safe='')}",
                params={"ref": target},
                timeout=TIMEOUT,
            ).ok
            actions.append({
                "action": "update" if exists else "create",
                "file_path": file_path,
                "content": content,
            })

        response = self.session.post(
            f"{self._project_url(project)}/repository/commits",
            json={
                "branch": branch,
                "start_branch": target,
                "commit_message": title,
                "actions": actions,
            },
            timeout=TIMEOUT,
        )
        check_response(response, f"a new branch in {project}")

        response = self.session.post(
            f"{self._project_url(project)}/merge_requests",
            json={
                "source_branch": branch,
                "target_branch": target,
                "title": title,
                "description": description,
                "remove_source_branch": True,
            },
            timeout=TIMEOUT,
        )
        check_response(response, f"a merge request in {project}")
        created = response.json()

        return {"url": created["web_url"], "number": created["iid"]}

    # --- Users ----------------------------------------------------------------

    def current_user(self) -> UserInfo:
        data = self._get(f"{self.base_url}/api/v4/user", "the current user").json()
        return UserInfo(
            id=str(data["id"]),
            username=data.get("username", ""),
            name=data.get("name") or data.get("username", ""),
        )

    def can_read(self, project: str) -> bool:
        try:
            self._get(self._project_url(project), project)
        except (AccessDenied, NotFound):
            return False

        return True
