"""The app: one local page that shows the BN and runs queries, by form or in words.

Standard library only. Endpoints:

    GET  /                        the page
    GET  /api/views               the schema and every compiled BN, each with its query menu
    POST /api/query               a probabilistic query -> its answer          (src/inference)
    POST /api/ask                 a question -> streamed query, computed, answer (src/llm)
    GET  /api/chat-status         whether a language-model backend is ready
    GET  /api/shakemap-events?network=<id>&min_magnitude=5
                                  USGS events with a ShakeMap near a network
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from src.builder import shakemap
from src.builder.load import RegistryError, load_network
from src.inference import engine
from src.llm import chat

from .views import view_data

PAGE = Path(__file__).with_name("page.html")

# A raw SDK traceback is no use on the page. Say what went wrong and what to do;
# the full text still goes to the server log.
_API_ERRORS = {
    "AuthenticationError": ("The Anthropic credential was rejected. Check ANTHROPIC_API_KEY, or run "
                            "ant auth status to see which credential the server picked up, then restart it."),
    "PermissionDeniedError": "That credential is not permitted to use this model.",
    "BadRequestError": ("The API rejected the request. A new account with no credit balance "
                        "fails this way; the server log carries the reason it gave."),
    "RateLimitError": "Rate limited by the API. Wait a moment and ask again.",
    "APIConnectionError": "Could not reach the Anthropic API. Check the network connection.",
    "APITimeoutError": "The API call timed out. Ask again, or use the query form.",
    "NotFoundError": "The configured model is not available to this credential.",
}


def _readable_api_error(exc: Exception) -> str:
    name = type(exc).__name__
    print(f"[bn] /api/ask failed: {name}: {exc}", flush=True)
    return _API_ERRORS.get(name, f"The request failed ({name}). See the server log for detail.")


_views: dict[str, Any] = {}
_views_lock = threading.Lock()


def views() -> dict[str, Any]:
    """Compiled once per server: the schema and instances do not change while it runs."""
    with _views_lock:
        if not _views:
            _views.update(view_data())
        return _views


class Handler(BaseHTTPRequestHandler):
    def _json(self, value: Any, status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_request(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        url = urlparse(self.path)
        if url.path in {"/", "/index.html"}:
            body = PAGE.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if url.path == "/api/views":
            return self._json(views())
        if url.path == "/api/chat-status":
            ok, reason = chat.available()
            return self._json({"available": ok, "reason": reason, "provider": chat.provider(),
                               "model": chat.model_name()})
        if url.path == "/api/shakemap-events":
            params = parse_qs(url.query)
            try:
                network = load_network((params.get("network") or [""])[0], engine.library())
                events = shakemap.search_events(network.rows("Node"),
                                                min_magnitude=float((params.get("min_magnitude") or [5])[0]))
            except RegistryError as exc:
                return self._json({"error": "; ".join(exc.problems)}, HTTPStatus.NOT_FOUND)
            except shakemap.ShakeMapError as exc:
                return self._json({"error": str(exc)}, HTTPStatus.BAD_GATEWAY)
            return self._json({"events": events})
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path == "/api/query":
            try:
                return self._json(engine.answer(self._read_request()))
            except (engine.QueryError, json.JSONDecodeError) as exc:
                return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        if path == "/api/ask":
            return self._ask()
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _ask(self) -> None:
        """Answer a question as a stream of events, one JSON object per line.

        The page shows each query before its values exist, so the reply cannot be
        one body written at the end: ``query`` goes out when the model picks it,
        ``computed`` (or ``refused``) when the engine returns, ``answer`` last.
        Headers are held back until the first event, so a request that fails
        before then still gets a real status code and a JSON error.
        """
        started = False

        def send(event: dict[str, Any]) -> None:
            nonlocal started
            try:
                if not started:
                    started = True
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
            except OSError:
                pass           # the reader went away; raising here would tell the model its query failed

        def fail(message: str, status: int) -> None:
            if started:
                send({"type": "error", "error": message})
            else:
                self._json({"error": message}, status)

        try:
            request = self._read_request()
            reply = chat.ask(request.get("assembly", ""), request.get("question", ""),
                             request.get("history") or [], defaults=request.get("defaults"), on_event=send)
            send({"type": "answer", **reply})
        except chat.ChatUnavailable as exc:
            fail(str(exc), HTTPStatus.SERVICE_UNAVAILABLE)
        except (engine.QueryError, json.JSONDecodeError) as exc:
            fail(str(exc), HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # an API failure should not take the server down
            fail(_readable_api_error(exc), HTTPStatus.BAD_GATEWAY)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[bn] {self.address_string()} {format % args}")


def make_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> None:
    server = make_server(host, port)
    url = f"http://{host}:{server.server_address[1]}"
    views()                               # compile everything now, so a broken instance fails at start
    ok, reason = chat.available()
    print(f"BN app: {url}")
    print(f"Language model: {chat.provider()} ({chat.model_name()}), " + ("ready" if ok else f"not available. {reason}"))
    if open_browser:
        webbrowser.open(url)
    if chat.provider() == "ollama":
        threading.Thread(target=lambda: print(f"[bn] {chat.preload_ollama()}", flush=True), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
