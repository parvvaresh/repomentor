"""GitHub REST API (github.com or GitHub Enterprise)."""

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

    if response.status_code == 403 and response.headers.get(
        "X-RateLimit-Remaining"
    ) == "0":
        raise SourceError(
            "GitHub's API rate limit was reached. Set GITHUB_TOKEN to raise it."
        )

    if response.status_code in (401, 403):
        raise AccessDenied(f"GitHub refused access to {what}.")

    if response.status_code == 404:
        raise NotFound(f"GitHub could not find {what}, or the token cannot see it.")

    raise SourceError(
        f"GitHub returned {response.status_code} for {what}: "
        f"{response.text[:200]}"
    )


class GitHubSource(RepositorySource):
    provider = "github"

    def __init__(
        self,
        api_url: str = "https://api.github.com",
        web_base_url: str = "https://github.com",
        token: str | None = None,
    ):
        self.api_url = api_url.rstrip("/")
        self.web_base_url = web_base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/vnd.github+json"
        self.session.headers["X-GitHub-Api-Version"] = "2022-11-28"

        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    # --- Helpers ------------------------------------------------------------

    def _repo_url(self, project: str) -> str:
        return f"{self.api_url}/repos/{project}"

    def _request(self, method: str, url: str, what: str, **kwargs):
        try:
            response = self.session.request(method, url, timeout=TIMEOUT, **kwargs)
        except requests.RequestException as error:
            raise SourceError(f"Could not reach GitHub: {error}") from error

        check_response(response, what)
        return response

    def _get(self, url: str, what: str, **params):
        return self._request("GET", url, what, params=params)

    def _paginate(self, url: str, what: str, limit: int | None = None, **params):
        count = 0
        page = 1

        while True:
            response = self._get(url, what, per_page=100, page=page, **params)
            items = response.json()

            for item in items:
                yield item
                count += 1

                if limit is not None and count >= limit:
                    return

            if 'rel="next"' not in response.headers.get("Link", ""):
                return

            page += 1

    def _default_branch(self, project: str) -> str:
        return self._get(self._repo_url(project), project).json()["default_branch"]

    # --- Files --------------------------------------------------------------

    def web_url(self, project: str) -> str:
        return f"{self.web_base_url}/{project}"

    def head_commit(self, project: str) -> str:
        branch = self._default_branch(project)
        return self._get(
            f"{self._repo_url(project)}/commits/{quote(branch, safe='')}",
            f"the {branch} branch of {project}",
        ).json()["sha"]

    def list_files(self, project: str, ref: str) -> list[FileEntry]:
        tree = self._get(
            f"{self._repo_url(project)}/git/trees/{ref}",
            f"the file tree of {project}",
            recursive="1",
        ).json()

        if tree.get("truncated"):
            raise SourceError(
                f"{project} has too many files for GitHub's tree API."
            )

        return [
            FileEntry(path=item["path"], blob_id=item["sha"])
            for item in tree["tree"]
            if item["type"] == "blob" and is_useful_file(item["path"])
        ]

    def read_file(self, project: str, file_path: str, ref: str) -> dict:
        data = self._get(
            f"{self._repo_url(project)}/contents/{quote(file_path)}",
            file_path,
            ref=ref,
        ).json()

        if data.get("encoding") == "base64":
            content = base64.b64decode(data["content"]).decode(
                "utf-8", errors="replace"
            )
        else:
            # Files over 1 MB come without content; fetch the raw blob
            content = self._request(
                "GET",
                f"{self._repo_url(project)}/git/blobs/{data['sha']}",
                file_path,
                headers={"Accept": "application/vnd.github.raw"},
            ).text

        return {
            "content": content,
            "blob_id": data["sha"],
            # Finding the last commit per file costs one request per file,
            # so links are pinned to the indexed commit instead.
            "last_commit_id": ref,
            "size": data.get("size", len(content)),
        }

    # --- History ------------------------------------------------------------

    def _commit(self, item: dict) -> dict:
        commit = item["commit"]
        message = commit.get("message") or ""

        return {
            "sha": item["sha"],
            "title": message.splitlines()[0] if message else "",
            "message": message,
            "author_name": (commit.get("author") or {}).get("name", ""),
            "author_email": (commit.get("author") or {}).get("email", ""),
            "date": (commit.get("author") or {}).get("date", ""),
            "url": item.get("html_url", ""),
        }

    def list_commits(self, project: str, limit: int) -> list[dict]:
        return [
            self._commit(item)
            for item in self._paginate(
                f"{self._repo_url(project)}/commits",
                f"commits of {project}",
                limit=limit,
            )
        ]

    def file_history(self, project: str, file_path: str, limit: int) -> list[dict]:
        return [
            self._commit(item)
            for item in self._paginate(
                f"{self._repo_url(project)}/commits",
                f"history of {file_path}",
                limit=limit,
                path=file_path,
            )
        ]

    def list_merge_requests(self, project: str, limit: int) -> list[dict]:
        merged = []

        for item in self._paginate(
            f"{self._repo_url(project)}/pulls",
            f"pull requests of {project}",
            state="closed",
            sort="updated",
            direction="desc",
        ):
            if not item.get("merged_at"):
                continue

            merged.append({
                "number": item["number"],
                "title": item.get("title") or "",
                "description": item.get("body") or "",
                "author": (item.get("user") or {}).get("login", ""),
                "merged_at": item["merged_at"],
                "url": item["html_url"],
            })

            if len(merged) >= limit:
                break

        return merged

    # --- Reviews and changes -----------------------------------------------

    def get_merge_request(self, project: str, number: int) -> dict:
        url = f"{self._repo_url(project)}/pulls/{number}"
        details = self._get(url, f"pull request #{number}").json()
        files = list(self._paginate(f"{url}/files", f"changes of #{number}"))

        return {
            "number": number,
            "title": details.get("title") or "",
            "description": details.get("body") or "",
            "author": (details.get("user") or {}).get("login", ""),
            "url": details["html_url"],
            "source_branch": details["head"]["ref"],
            "target_branch": details["base"]["ref"],
            "changes": [
                {
                    "path": item["filename"],
                    "old_path": item.get("previous_filename") or item["filename"],
                    "diff": item.get("patch") or "",
                    "new_file": item["status"] == "added",
                    "deleted_file": item["status"] == "removed",
                }
                for item in files
            ],
        }

    def create_merge_request(self, project, branch, title, description, files):
        target = self._default_branch(project)
        base_sha = self._get(
            f"{self._repo_url(project)}/git/ref/heads/{quote(target, safe='')}",
            f"the {target} branch",
        ).json()["object"]["sha"]

        self._request(
            "POST",
            f"{self._repo_url(project)}/git/refs",
            f"a new branch in {project}",
            json={"ref": f"refs/heads/{branch}", "sha": base_sha},
        )

        for file_path, content in files.items():
            body = {
                "message": title,
                "content": base64.b64encode(content.encode()).decode(),
                "branch": branch,
            }

            try:
                existing = self._get(
                    f"{self._repo_url(project)}/contents/{quote(file_path)}",
                    file_path,
                    ref=branch,
                ).json()
                body["sha"] = existing["sha"]
            except NotFound:
                pass

            self._request(
                "PUT",
                f"{self._repo_url(project)}/contents/{quote(file_path)}",
                f"{file_path} on {branch}",
                json=body,
            )

        created = self._request(
            "POST",
            f"{self._repo_url(project)}/pulls",
            f"a pull request in {project}",
            json={
                "title": title,
                "body": description,
                "head": branch,
                "base": target,
            },
        ).json()

        return {"url": created["html_url"], "number": created["number"]}

    # --- Users ----------------------------------------------------------------

    def current_user(self) -> UserInfo:
        data = self._get(f"{self.api_url}/user", "the current user").json()
        return UserInfo(
            id=str(data["id"]),
            username=data.get("login", ""),
            name=data.get("name") or data.get("login", ""),
        )

    def can_read(self, project: str) -> bool:
        try:
            self._get(self._repo_url(project), project)
        except (AccessDenied, NotFound):
            return False

        return True
