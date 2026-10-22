"""quakenet: one local page that runs an example network's BN, by form or in words.

Standard library only. Endpoints:

    GET  /                        quakenet: scenario, map, results, BN, questions in words
    GET  /bn                      the BN drawing (also embedded in quakenet's BN tab)
    GET  /api/examples            every example assembly: id, label, network
    GET  /api/example?assembly=<id>
                                  one example: its network's rows, its query menu, its stored rules
    GET  /api/views               the schema and every compiled BN, each with its query menu
    POST /api/query               a probabilistic query -> its answer          (src/inference)
    POST /api/ask                 a question -> streamed query, computed, answer (src/llm)
    GET  /api/chat-status         whether a language-model backend is ready
    GET  /api/shakemap-events?network=<id>&min_magnitude=5
                                  USGS events with a ShakeMap near a network
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from src.builder import shakemap
from src.builder.bn_builder import attach_rules, compile_path, model_dir, tbn_data
from src.builder.load import (ROOT, RegistryError, assembly_id_of, assembly_path, assembly_paths, load_assembly,
                              load_network)
from src.inference import engine
from src.llm import chat

from .views import view_data

PAGE = Path(__file__).with_name("quakenet.html")
BN_PAGE = Path(__file__).with_name("page.html")

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


def examples() -> list[dict[str, Any]]:
    out = []
    for path in assembly_paths():
        a = load_assembly(path)
        out.append({"id": assembly_id_of(path), "label": a["label"], "network": a["network"]})
    return out


def example(assembly_id: str) -> dict[str, Any]:
    """What quakenet needs to draw one example and its controls, before anything is sampled."""
    path = assembly_path(assembly_id)
    compiled = compile_path(path, engine.library())
    network = compiled.network
    data = tbn_data(compiled)
    systems = [s for s in data["probabilities"].values() if s["role"] == "system"]
    stored = attach_rules(data, model_dir(path))
    return {
        "assembly": compiled.assembly, "menu": engine.describe(assembly_id),
        "network": {"id": network.id, "title": network.title, "assumptions": network.assumptions,
                    "source": network.source, "tables": network.tables},
        "functions": [compiled.bindings[i].function["id"] for i in compiled.order],
        "objects": len(data["probabilities"]), "variables": len(data["variables"]),
        # Every BN node and its parents, for the layered drawing; one joint object per stochastic Joint binding.
        "nodes": [{k: n[k] for k in ("id", "quantity", "entity", "row", "function", "parameter_set", "parents")}
                  for n in compiled.nodes()],
        "joint": sorted({s["function"] for s in data["probabilities"].values() if len(s["childs"]) > 1}),
        "rules": {"stored": stored, "origins": sum(len(s["rows"]) for s in systems)},
    }


# Rule extractions running in the background, one per (assembly, threshold). A separate process, so a
# long extraction neither holds up queries nor disturbs the random state their seeds fix.
_extractions: dict[tuple[str, float], subprocess.Popen] = {}
_extractions_lock = threading.Lock()


def extract_in_background(assembly: str, threshold: float, origins: list[str]) -> str:
    """Start ``python -m src rules`` for origins without rules of their own at this threshold; say so."""
    key = (assembly, threshold)
    with _extractions_lock:
        job = _extractions.get(key)
        if job is not None and job.poll() is None:
            return (f"rsr rules for target_add_h={threshold:g} are still being extracted in the background; "
                    "later runs will use them.")
        log = ROOT / ".cache" / f"rules-{assembly.replace('/', '_')}-{threshold:g}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as out:
            _extractions[key] = subprocess.Popen(
                [sys.executable, "-m", "src", "rules", assembly, "--target-add-h", repr(threshold), "--origins", *origins],
                cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    print(f"[quakenet] extracting rsr rules for {assembly} at target_add_h={threshold:g}: {', '.join(origins)}", flush=True)
    return (f"Extracting rsr rules for target_add_h={threshold:g} in the background, for {len(origins)} origin(s); "
            f"later runs will use them (log: .cache/{log.name}).")


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

    def _page(self, path: Path, content_type: str = "text/html; charset=utf-8") -> None:
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        url = urlparse(self.path)
        if url.path == "/api/proposals":
            from src import proposals
            return self._json({"proposals": proposals.list_proposals()})
        if url.path == "/api/scenarios":
            from src import proposals
            return self._json({"scenarios": proposals.scenarios((parse_qs(url.query).get("assembly") or [""])[0])})
        if url.path in {"/", "/index.html"}:
            return self._page(PAGE)
        if url.path == "/ask.js":
            return self._page(PAGE.with_name("ask.js"), "text/javascript; charset=utf-8")
        if url.path == "/bn":
            return self._page(BN_PAGE)
        if url.path == "/api/examples":
            return self._json({"examples": examples()})
        if url.path == "/api/example":
            try:
                return self._json(example((parse_qs(url.query).get("assembly") or [""])[0]))
            except (RegistryError, engine.QueryError) as exc:
                problems = exc.problems if isinstance(exc, RegistryError) else [str(exc)]
                return self._json({"error": "; ".join(problems)}, HTTPStatus.NOT_FOUND)
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
        if path in {"/api/proposals", "/api/proposals/activate", "/api/review"}:
            try:
                from src import proposals
                body = self._read_request()
                if path == "/api/review":
                    from .reviews import decide
                    decide(body.get("token"), body.get("approved"))
                    return self._json({"ok": True})
                if path == "/api/proposals":
                    return self._json(proposals.draft(body))
                result = proposals.activate(body.get("id", ""), body.get("reviewer"), body.get("sha256"))
                with _views_lock:
                    _views.clear()
                return self._json(result)
            except (engine.QueryError, ValueError, TypeError, FileExistsError) as exc:
                return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        if path == "/api/query":
            try:
                query, notes = self._read_request(), []
                # Started before sampling, from what is stored: a slow run does not hold the extraction up.
                result = engine.answer(query, on_missing_rules=lambda threshold, origins: notes.append(
                    extract_in_background(query.get("assembly", ""), threshold, origins)), record={"source": "page"})
                result["warnings"] += notes
                return self._json(result)
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
            from .reviews import wait_for_review
            reply = chat.ask(request.get("assembly", ""), request.get("question", ""),
                             request.get("history") or [], defaults=request.get("defaults"), on_event=send,
                             review=(lambda query: wait_for_review(query, send)) if request.get("review") else None)
            send({"type": "answer", **reply})
        except chat.ChatUnavailable as exc:
            fail(str(exc), HTTPStatus.SERVICE_UNAVAILABLE)
        except (engine.QueryError, json.JSONDecodeError) as exc:
            fail(str(exc), HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # an API failure should not take the server down
            fail(_readable_api_error(exc), HTTPStatus.BAD_GATEWAY)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[bn] {self.address_string()} {format % args}")


class _Server(ThreadingHTTPServer):
    # On Windows, SO_REUSEADDR lets a second server bind a port that one is already listening on,
    # and requests then reach either: an old server keeps answering beside a new one. Refuse instead.
    allow_reuse_address = sys.platform != "win32"


def make_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return _Server((host, port), Handler)


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False, example: str | None = None) -> None:
    try:
        server = make_server(host, port)
    except OSError as exc:
        raise SystemExit(f"Port {port} is already in use, probably by another quakenet server. Stop that one "
                         f"(Ctrl+C in its window), or start this one with --port {port + 1}. ({exc})") from None
    url = f"http://{host}:{server.server_address[1]}" + (f"/#example={example}" if example else "")
    views()                               # compile everything now, so a broken instance fails at start
    ok, reason = chat.available()
    print(f"quakenet: {url}", flush=True)
    print(f"Language model: {chat.provider()} ({chat.model_name()}), " + ("ready" if ok else f"not available. {reason}"),
          flush=True)
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
