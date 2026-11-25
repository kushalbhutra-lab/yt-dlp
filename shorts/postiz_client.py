"""
Minimal Postiz public-API client (stdlib only, no extra dependencies).

Mirrors the official Postiz SDK (apps/sdk in gitroomhq/postiz-app):

    base URL : https://api.postiz.com for Postiz Cloud,
               https://<your-domain>/api for a self-hosted instance
    auth     : Authorization: <api key>   (raw key, no "Bearer" prefix)
    routes   : /public/v1/{upload,integrations,find-slot/:id,posts}

Endpoints used here
-------------------
GET  /public/v1/integrations    -> [{id, name, identifier, disabled, ...}]
POST /public/v1/upload          -> {id, name, path, thumbnail, alt, status}
GET  /public/v1/find-slot/:id   -> {date: "..."}
POST /public/v1/posts           -> created post(s)
"""

from __future__ import annotations

import json
import mimetypes
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "https://api.postiz.com"
TIMEOUT = 300  # video uploads are slow


class PostizError(RuntimeError):
    """Raised when the Postiz API rejects a request."""


class PostizClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL) -> None:
        if not api_key:
            raise PostizError("a Postiz API key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    # -------------------------------------------------------------- plumbing #

    def _url(self, path: str) -> str:
        return f"{self.base_url}/public/v1/{path.lstrip('/')}"

    def _send(self, request: urllib.request.Request) -> Any:
        request.add_header("Authorization", self.api_key)
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:  # Postiz errors come back as {"msg": "..."}
                parsed = json.loads(detail)
                detail = parsed.get("msg") or parsed.get("message") or detail
            except Exception:
                pass
            hint = ""
            if exc.code == 401:
                hint = " (check --postiz-key; the header takes the raw key)"
            elif exc.code == 404:
                hint = f" (check --postiz-url; tried {request.full_url})"
            raise PostizError(f"HTTP {exc.code} from Postiz{hint}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise PostizError(
                f"could not reach Postiz at {self.base_url}: {exc.reason}"
            ) from exc

        if not raw.strip():
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = self._url(path)
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        return self._send(urllib.request.Request(url, method="GET"))

    def _post_json(self, path: str, payload: dict) -> Any:
        request = urllib.request.Request(
            self._url(path),
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
        )
        request.add_header("Content-Type", "application/json")
        return self._send(request)

    # ------------------------------------------------------------- endpoints #

    def integrations(self, group: str | None = None) -> list[dict]:
        result = self._get("integrations", {"group": group} if group else None)
        return result if isinstance(result, list) else []

    def upload(self, file_path: Path) -> dict:
        """Upload a media file as multipart/form-data under the field 'file'."""
        file_path = Path(file_path)
        if not file_path.is_file():
            raise PostizError(f"file not found: {file_path}")

        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        boundary = f"----postiz{uuid.uuid4().hex}"
        preamble = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
        epilogue = f"\r\n--{boundary}--\r\n".encode()
        body = preamble + file_path.read_bytes() + epilogue

        request = urllib.request.Request(self._url("upload"), data=body, method="POST")
        request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        request.add_header("Content-Length", str(len(body)))

        result = self._send(request)
        if not isinstance(result, dict) or "id" not in result or "path" not in result:
            raise PostizError(f"unexpected upload response: {result!r}")
        return result

    def find_slot(self, integration_id: str) -> str | None:
        result = self._get(f"find-slot/{urllib.parse.quote(integration_id)}")
        if isinstance(result, dict):
            date = result.get("date")
            return str(date) if date else None
        return None

    def create_post(
        self,
        *,
        integration_ids: list[str],
        content: str,
        media: list[dict],
        date_iso: str,
        post_type: str = "draft",
        settings: dict | None = None,
        short_link: bool = False,
        tags: list[dict] | None = None,
    ) -> Any:
        """
        Create one post that fans out to several channels.

        post_type: 'draft' | 'schedule' | 'now' | 'update'. 'draft' skips the
        per-provider settings validation, which is why it is the safe default:
        channels such as YouTube and TikTok reject a non-draft post that has no
        title or privacy settings of their own.
        """
        if post_type not in {"draft", "schedule", "now", "update"}:
            raise PostizError(f"invalid post type: {post_type}")
        if not integration_ids:
            raise PostizError("no Postiz channels selected")

        images = [
            {"id": item["id"], "path": item["path"]}
            for item in media
            if item.get("id") and item.get("path")
        ]
        payload = {
            "type": post_type,
            "shortLink": short_link,
            "date": date_iso,
            "tags": tags or [],
            "posts": [
                {
                    "integration": {"id": integration_id},
                    "value": [{"content": content, "image": images}],
                    "settings": settings or {},
                }
                for integration_id in integration_ids
            ],
        }
        return self._post_json("posts", payload)
