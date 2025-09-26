"""Test callback server used by the async inference end-to-end suite."""

from __future__ import annotations

import asyncio
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
import uvicorn

CALLBACK_SECRET_HEADER = "X-Callback-Secret"


@dataclass
class CallbackStore:
    """In-memory record of callbacks persisted to disk for inspection."""

    root: Path = field(default_factory=lambda: Path("mock_callbacks"))

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.image_dir = self.root / "images"
        self.reset()

    def reset(self) -> None:
        """Clear previous callbacks and recreate storage folders."""
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.callbacks: List[Dict[str, Any]] = []

    def record(self, data: Dict[str, Any]) -> None:
        self.callbacks.append(data)

    def all(self) -> List[Dict[str, Any]]:
        return list(self.callbacks)


def create_callback_app(store: Optional[CallbackStore] = None) -> FastAPI:
    """Return a FastAPI application that mimics the production callback target."""

    store = store or CallbackStore()
    app = FastAPI(title="Async Inference Callback Server")
    app.state.store = store

    @app.post("/callback")
    async def receive_callback(
        request: Request,
        job_id: str = Form(...),
        status: str = Form(...),
        completed_at: str = Form(...),
        image_url: Optional[str] = Form(None),
        error: Optional[str] = Form(None),
        image: UploadFile = File(None),
    ) -> Dict[str, Any]:
        received_at = datetime.now(timezone.utc).isoformat()

        entry: Dict[str, Any] = {
            "job_id": job_id,
            "status": status,
            "completed_at": completed_at,
            "received_at": received_at,
            "image_url": image_url,
            "provided_secret": request.headers.get(CALLBACK_SECRET_HEADER),
            "has_image": image is not None,
            "error": error,
        }

        if image:
            content = await image.read()
            image_path = store.image_dir / f"{job_id}.png"
            with open(image_path, "wb") as handle:
                handle.write(content)
            entry.update({
                "image_size": len(content),
                "saved_path": str(image_path),
            })

        store.record(entry)
        return {"status": "success", "message": f"Callback received for job {job_id}"}

    @app.get("/callbacks")
    async def list_callbacks() -> Dict[str, Any]:
        return {"callbacks": app.state.store.all()}

    @app.get("/")
    async def dashboard() -> HTMLResponse:
        html = """
        <html>
            <head><title>Async Inference Callback Dashboard</title></head>
            <body>
                <h1>Async Inference Callback Dashboard</h1>
                <div id="callbacks">Loading...</div>
                <script>
                    async function loadCallbacks() {
                        const response = await fetch('/callbacks');
                        const data = await response.json();
                        const container = document.getElementById('callbacks');
                        container.innerHTML = '<h2>Received Callbacks:</h2>';
                        if (data.callbacks.length === 0) {
                            container.innerHTML += '<p>No callbacks received yet.</p>';
                        } else {
                            data.callbacks.forEach(cb => {
                                container.innerHTML += `
                                    <div style="border: 1px solid #ccc; margin: 8px; padding: 8px;">
                                        <h3>Job ID: ${cb.job_id || 'unknown'}</h3>
                                        <p>Status: ${cb.status}</p>
                                        <p>Completed At: ${cb.completed_at}</p>
                                        <p>Received At: ${cb.received_at}</p>
                                        <p>Secret Header: ${cb.provided_secret || '—'}</p>
                                        ${cb.image_url ? `<p>Image URL: ${cb.image_url}</p>` : ''}
                                        ${cb.has_image ? `<p>✅ Image attached (${cb.image_size || 0} bytes)</p>` : '<p>❌ No image payload</p>'}
                                    </div>
                                `;
                            });
                        }
                    }
                    loadCallbacks();
                    setInterval(loadCallbacks, 2000);
                </script>
            </body>
        </html>
        """
        return HTMLResponse(content=html)

    return app


class CallbackServer:
    """Utility to run the callback FastAPI app in a background thread."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8092,
        probe_host: Optional[str] = None,
        storage_dir: Path | str = Path("mock_callbacks"),
        log_level: str = "warning",
        ready_probe_timeout: float = 10.0,
    ) -> None:
        self.host = host
        self.port = port
        self.probe_host = probe_host or host
        self.storage_dir = Path(storage_dir)
        self.log_level = log_level
        self.ready_probe_timeout = ready_probe_timeout
        self.store = CallbackStore(self.storage_dir)
        self.app = create_callback_app(self.store)
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[uvicorn.Server] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.probe_host}:{self.port}"

    def _build_server(self) -> uvicorn.Server:
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level=self.log_level,
        )
        return uvicorn.Server(config)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self.store.reset()
        self._server = self._build_server()

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._server.serve())

        self._thread = threading.Thread(target=_run, name="callback-server", daemon=True)
        self._thread.start()
        self._wait_for_ready()

    def _wait_for_ready(self) -> None:
        deadline = time.time() + self.ready_probe_timeout
        url = f"{self.base_url}/callbacks"
        while time.time() < deadline:
            try:
                response = requests.get(url, timeout=0.5)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                time.sleep(0.1)
            else:
                time.sleep(0.1)
        raise RuntimeError("Callback server failed to start before timeout")

    def stop(self) -> None:
        if not self._thread:
            return
        if self._server:
            self._server.should_exit = True
        self._thread.join(timeout=5.0)
        self._thread = None
        self._server = None

    def callbacks(self) -> List[Dict[str, Any]]:
        return self.store.all()


def main() -> None:
    """CLI entry-point for running the callback server standalone."""
    server = CallbackServer(host="0.0.0.0")
    try:
        server.start()
        print(f"Callback server listening on {server.base_url}")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down callback server...")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
