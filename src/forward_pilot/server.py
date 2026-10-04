"""Local HTTP server for the forward-inference pilot UI.

Standard-library only, with one optional exception: the natural-language panel
needs the `anthropic` package and a credential. Both are checked at request time,
so the pilot runs unchanged when neither is present.
"""

from __future__ import annotations

import json
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import chat
from .engine import ForwardQueryError, run_forward_query, tbn_available
from .package import DEFAULT_OUTPUT, assemble_all, load_package


UI_ROOT = Path(__file__).parent / "ui"

# A raw SDK traceback is no use to someone being shown the pilot. Say what went
# wrong and what to do; the full text still goes to the server log.
_API_ERRORS = {
    "AuthenticationError": (
        "The Anthropic credential was rejected. Check ANTHROPIC_API_KEY, or run "
        "ant auth status to see which credential the server picked up, then restart it."
    ),
    "PermissionDeniedError": "That credential is not permitted to use this model.",
    "BadRequestError": (
        "The API rejected the request. A new account with no credit balance "
        "fails this way; the server log carries the reason it gave."
    ),
    "RateLimitError": "Rate limited by the API. Wait a moment and ask again.",
    "APIConnectionError": "Could not reach the Anthropic API. Check the network connection.",
    "APITimeoutError": "The API call timed out. Ask again, or use the controls directly.",
    "NotFoundError": "The configured model is not available to this credential.",
}


def _readable_api_error(exc: Exception) -> str:
    name = type(exc).__name__
    # flush: stdout is block-buffered when the server is run with output
    # redirected to a file, so without this the diagnostic can be lost entirely
    # if the process is stopped before the buffer fills.
    print(f"[forward-pilot] /api/ask failed: {name}: {exc}", flush=True)
    return _API_ERRORS.get(name, f"The request failed ({name}). See the server log for detail.")


class PilotHandler(BaseHTTPRequestHandler):
    packages_root = DEFAULT_OUTPUT

    def _json(self, value: Any, status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _package_ids(self) -> list[str]:
        if not self.packages_root.exists():
            return []
        return sorted(
            path.name
            for path in self.packages_root.iterdir()
            if path.is_dir() and (path / "network.json").is_file()
        )

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path == "/api/catalog":
            items = []
            for package_id in self._package_ids():
                network = load_package(self.packages_root / package_id)
                items.append(
                    {
                        "case_id": package_id,
                        "title": network["title"],
                        "description": network["description"],
                        "nodes": len(network["nodes"]),
                        "edges": len(network["edges"]),
                        "directed": network["directed"],
                        "hazard_models": network["hazard_models"],
                    }
                )
            return self._json({"packages": items, "inference_scope": "forward_only"})
        if path.startswith("/api/package/"):
            package_id = path.removeprefix("/api/package/")
            if package_id not in self._package_ids():
                return self._json({"error": "package not found"}, HTTPStatus.NOT_FOUND)
            network = load_package(self.packages_root / package_id)
            spec = json.loads((self.packages_root / package_id / "bn_spec.json").read_text(encoding="utf-8"))
            return self._json({"network": network, "bn_spec": spec})
        if path.startswith("/api/bn-graph/"):
            package_id = path.removeprefix("/api/bn-graph/")
            if package_id not in self._package_ids():
                return self._json({"error": "package not found"}, HTTPStatus.NOT_FOUND)
            if not tbn_available():
                return self._json(
                    {"error": "the instance view needs tbnpy and torch installed"},
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
            from .tbn.graph import structure

            query = parse_qs(urlparse(self.path).query)
            limit = int((query.get("edges") or ["3"])[0])
            network = load_package(self.packages_root / package_id)
            return self._json(structure(network, edge_limit=max(1, min(6, limit))))
        if path == "/api/chat-status":
            ok, reason = chat.available()
            return self._json({"available": ok, "reason": reason, "model": chat.MODEL})
        if path in {"/", "/index.html"}:
            return self._static("index.html", "text/html; charset=utf-8")
        if path == "/app.js":
            return self._static("app.js", "text/javascript; charset=utf-8")
        if path == "/style.css":
            return self._static("style.css", "text/css; charset=utf-8")
        self.send_error(HTTPStatus.NOT_FOUND)

    def _static(self, filename: str, content_type: str) -> None:
        body = (UI_ROOT / filename).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_request(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _network_for(self, package_id: str):
        if package_id not in self._package_ids():
            raise KeyError(f"package not found: {package_id}")
        return load_package(self.packages_root / package_id)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path == "/api/query":
            try:
                request = self._read_request()
                network = self._network_for(request.pop("package_id"))
                self._json(run_forward_query(network, request))
            except KeyError as exc:
                self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            except (ForwardQueryError, ValueError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        if path == "/api/ask":
            try:
                request = self._read_request()
                network = self._network_for(request.get("package_id"))
                self._json(chat.ask(
                    network,
                    request.get("question", ""),
                    request.get("history") or [],
                ))
            except KeyError as exc:
                self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            except chat.ChatUnavailable as exc:
                self._json({"error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
            except json.JSONDecodeError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # an API failure should not take the server down
                self._json({"error": _readable_api_error(exc)}, HTTPStatus.BAD_GATEWAY)
            return

        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[forward-pilot] {self.address_string()} {format % args}")


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    packages_root: Path = DEFAULT_OUTPUT,
    open_browser: bool = False,
) -> None:
    if not packages_root.exists() or not any(packages_root.iterdir()):
        assemble_all(packages_root)
    handler = type("ConfiguredPilotHandler", (PilotHandler,), {"packages_root": packages_root})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Forward BN pilot: http://{host}:{port}")
    print(f"Packages: {packages_root}")
    if open_browser:
        webbrowser.open(f"http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
