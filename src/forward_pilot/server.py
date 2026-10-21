"""Local HTTP server for the forward-inference pilot UI.

Standard-library only. The natural-language panel needs a model backend: a local
Ollama server by default, or the `anthropic` package and a credential when
FORWARD_PILOT_LLM=anthropic. Either is checked at request time, so the pilot runs
unchanged when neither is present.
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


RECENT_DAYS = 30
RECENT_TTL = 60.0                 # seconds; the page asks every 2 minutes, USGS's feed updates every minute
_recent_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}


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

    def _recent_earthquakes(self) -> list[dict[str, Any]]:
        """Events of the last ``RECENT_DAYS`` with a ShakeMap around any package,
        newest first, each with the packages it concerns."""
        import time
        from datetime import datetime, timedelta, timezone

        from . import shakemap
        from .feedwatch import DEFAULT_MIN_MAGNITUDE

        key = str(self.packages_root)
        hit = _recent_cache.get(key)
        if hit and time.monotonic() - hit[0] < RECENT_TTL:
            return hit[1]
        start = (datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)).strftime("%Y-%m-%d")
        source = shakemap.ComcatSource()
        events: dict[str, dict[str, Any]] = {}
        for package_id in self._package_ids():
            network = load_package(self.packages_root / package_id)
            if "shakemap" not in network.get("hazard_models", ()):
                continue
            try:
                bbox = shakemap.network_bbox(network)
            except shakemap.ShakeMapError:
                continue                  # offers ShakeMaps but has no coordinates
            for event in source.search(bbox, start=start, min_magnitude=DEFAULT_MIN_MAGNITUDE):
                events.setdefault(event["event_id"], {**event, "packages": []})["packages"].append(package_id)
        found = sorted(events.values(), key=lambda e: e.get("time") or "", reverse=True)
        _recent_cache[key] = (time.monotonic(), found)
        return found

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
        if path.startswith("/api/shakemap-events/"):
            package_id = path.removeprefix("/api/shakemap-events/")
            if package_id not in self._package_ids():
                return self._json({"error": "package not found"}, HTTPStatus.NOT_FOUND)
            from . import shakemap

            query = parse_qs(urlparse(self.path).query)
            try:
                source = shakemap.get_source((query.get("source") or [None])[0])
                from .feedwatch import DEFAULT_MIN_MAGNITUDE

                # The same floor as the watcher, so anything it emails can also be picked here.
                events = shakemap.search_events(
                    load_package(self.packages_root / package_id), source.name,
                    min_magnitude=float((query.get("min_magnitude") or [DEFAULT_MIN_MAGNITUDE])[0]),
                )
            except shakemap.ShakeMapError as exc:
                return self._json({"error": str(exc)}, HTTPStatus.BAD_GATEWAY)
            return self._json({"source": source.name, "events": events})
        if path == "/api/recent-earthquakes":
            # Straight from USGS, so the page sees new events and aftershocks without the watcher.
            from . import shakemap

            try:
                return self._json({"days": RECENT_DAYS, "events": self._recent_earthquakes()})
            except shakemap.ShakeMapError as exc:
                return self._json({"error": str(exc)}, HTTPStatus.BAD_GATEWAY)
        if path.startswith("/api/shakemap-version/"):
            from . import shakemap

            event_id = path.removeprefix("/api/shakemap-version/")
            try:
                version = shakemap.ComcatSource().current_version(event_id)
            except shakemap.ShakeMapError as exc:
                return self._json({"error": str(exc)}, HTTPStatus.BAD_GATEWAY)
            return self._json({"event_id": event_id, "version": version})
        if path == "/api/chat-status":
            ok, reason = chat.available()
            return self._json({"available": ok, "reason": reason,
                               "provider": chat.provider(), "model": chat.model_name()})
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
                result = run_forward_query(network, request)
                hazard = request.get("hazard") or {}
                if hazard.get("model") == "shakemap":
                    # Which version ran, so the page can tell when USGS has revised it.
                    # The grid is already loaded: this reads the in-process cache.
                    from . import shakemap

                    grid = shakemap.load(hazard["event_id"], hazard.get("source"), hazard.get("version"))
                    result["shakemap"] = {"event_id": grid.event_id, "version": shakemap._int(grid.version)}
                self._json(result)
            except KeyError as exc:
                self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            except (ForwardQueryError, ValueError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        if path == "/api/ask":
            return self._ask()

        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _ask(self) -> None:
        """Answer a question as a stream of events, one JSON object per line.

        The page has to show the probabilistic query before the value exists, so
        the reply cannot be one body written at the end: ``query`` goes out when
        the agent picks the scenario, ``computed`` when the engine returns, and
        ``answer`` last. The response has no Content-Length and ends when the
        connection closes, which HTTP/1.0 -- this handler's default -- allows.

        Headers are held back until the first event. A request that fails before
        then (unknown package, no credential, a rejected key) still gets a real
        status code and the usual JSON error; only a failure after the stream has
        started has to travel inside it, as an ``error`` event.
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
                # The reader went away. This runs inside the tool call, so letting
                # it raise would be reported to the agent as the query having failed.
                pass

        def fail(message: str, status: int) -> None:
            if started:
                send({"type": "error", "error": message})
            else:
                self._json({"error": message}, status)

        try:
            request = self._read_request()
            network = self._network_for(request.get("package_id"))
            reply = chat.ask(
                network,
                request.get("question", ""),
                request.get("history") or [],
                on_event=send,
                scenario=request.get("scenario"),
            )
            send({"type": "answer", **reply})
        except KeyError as exc:
            fail(str(exc), HTTPStatus.NOT_FOUND)
        except chat.ChatUnavailable as exc:
            fail(str(exc), HTTPStatus.SERVICE_UNAVAILABLE)
        except json.JSONDecodeError as exc:
            fail(str(exc), HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # an API failure should not take the server down
            fail(_readable_api_error(exc), HTTPStatus.BAD_GATEWAY)

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
    ok, reason = chat.available()
    print(f"Chat: {chat.provider()} ({chat.model_name()}), " + ("ready" if ok else f"not available. {reason}"))
    if open_browser:
        webbrowser.open(f"http://{host}:{port}")
    # Load the local model in the background, so the first question does not
    # also pay for reading several GB from disk. No-op for hosted backends.
    import threading

    threading.Thread(target=lambda: print(f"[forward-pilot] {chat.preload_ollama()}", flush=True)
                     if chat.provider() == "ollama" else None, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
