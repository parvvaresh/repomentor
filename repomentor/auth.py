"""
Per-user access control for multi-user deployments.

With AUTH_MODE=none (the default) RepoMentor is a single-developer tool:
everyone who can reach the server sees every repository.

With AUTH_MODE=token, each user signs in with their own GitLab or GitHub
personal access token. The server never stores the token; the browser
sends it with each request. RepoMentor then:

- identifies the user with the host's "current user" endpoint;
- lists only repositories that the user's token can read on the host, so
  host permissions (private projects, groups, revoked membership) apply
  to RepoMentor too;
- keeps each user's chats private to them.

Indexing itself still uses the server's read token, so a repository is
downloaded once and shared by everyone allowed to read it. Results are
cached for ACCESS_CACHE_SECONDS to avoid one host request per API call.
"""

import hashlib
import threading
import time
from dataclasses import dataclass

from . import config
from .sources import SourceError, get_source


@dataclass(frozen=True)
class User:
    id: str
    provider: str | None
    username: str
    name: str
    token: str | None = None


LOCAL_USER = User(id="local", provider=None, username="local", name="Local user")


class AuthError(Exception):
    pass


class AccessControl:
    def __init__(self):
        self._users: dict[str, tuple[float, User]] = {}
        self._access: dict[tuple[str, str, str], tuple[float, bool]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return config.AUTH_MODE == "token"

    def _fresh(self, stored_at: float) -> bool:
        return time.monotonic() - stored_at < config.ACCESS_CACHE_SECONDS

    def authenticate(self, provider: str | None, token: str | None) -> User:
        if not self.enabled:
            return LOCAL_USER

        if not token or provider not in ("gitlab", "github"):
            raise AuthError("Sign in with a GitLab or GitHub access token.")

        key = hashlib.sha256(f"{provider}:{token}".encode()).hexdigest()

        with self._lock:
            cached = self._users.get(key)

        if cached and self._fresh(cached[0]):
            return cached[1]

        try:
            info = get_source(provider, token).current_user()
        except SourceError as error:
            raise AuthError(f"The {provider} token was rejected: {error}") from error

        user = User(
            id=f"{provider}:{info.id}",
            provider=provider,
            username=info.username,
            name=info.name,
            token=token,
        )

        with self._lock:
            self._users[key] = (time.monotonic(), user)

        return user

    def can_read(self, user: User, provider: str, project_path: str) -> bool:
        if not self.enabled:
            return True

        # A token from one host says nothing about projects on another,
        # and local checkouts have no owner to ask.
        if user.provider != provider:
            return False

        key = (user.id, provider, project_path)

        with self._lock:
            cached = self._access.get(key)

        if cached and self._fresh(cached[0]):
            return cached[1]

        try:
            allowed = get_source(provider, user.token).can_read(project_path)
        except SourceError:
            allowed = False

        with self._lock:
            self._access[key] = (time.monotonic(), allowed)

        return allowed

    def forget(self) -> None:
        with self._lock:
            self._users.clear()
            self._access.clear()


access = AccessControl()
