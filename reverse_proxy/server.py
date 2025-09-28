"""Main reverse proxy server for Bittensor subnet miner."""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

import httpx
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from loguru import logger

from reverse_proxy.config import Config, load_config
from reverse_proxy.epistula import EpistulaVerifier

CONFIG: Optional[Config] = None
epistula_verifier: Optional[EpistulaVerifier] = None

AUTH_SCHEME = "Epistula"
SIGNATURE_HEADER = "Epistula-Signature"
REQUIRED_EP_HEADERS = {
    "timestamp": "Epistula-Timestamp",
    "uuid_str": "Epistula-UUID",
    "signed_by": "Epistula-Signed-By",
    "signed_for": "Epistula-Signed-For",
}
_BODY_CACHE_ATTR = "_cached_body"
_EP_CONTEXT_ATTR = "epistula_context"


@dataclass(frozen=True)
class EpistulaRequestContext:
    """Collected Epistula authentication headers and body for a request."""

    signature: str
    timestamp: str
    uuid_str: str
    signed_by: str
    signed_for: str
    body: bytes

    @classmethod
    async def from_request(cls, request: Request) -> "EpistulaRequestContext":
        cached_context = getattr(request.state, _EP_CONTEXT_ATTR, None)
        if cached_context is not None:
            return cached_context

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Authorization header required")

        try:
            scheme, value = auth_header.split(" ", 1)
        except ValueError:
            raise HTTPException(status_code=401, detail="Invalid authorization header format")

        if scheme.strip() != AUTH_SCHEME:
            raise HTTPException(status_code=401, detail="Invalid authorization scheme")

        header_signature = value.strip()
        signature_header = request.headers.get(SIGNATURE_HEADER)
        if signature_header:
            signature_value = signature_header.strip()
            if signature_value != header_signature:
                raise HTTPException(status_code=401, detail="Authorization signature mismatch")
        else:
            signature_value = header_signature

        if not signature_value:
            raise HTTPException(status_code=401, detail="Missing Epistula authentication signature")

        if not signature_value.startswith("0x"):
            signature_value = f"0x{signature_value}"

        missing_headers = []
        header_values = {}
        for field, header_name in REQUIRED_EP_HEADERS.items():
            raw_value = request.headers.get(header_name)
            if raw_value is None or not raw_value.strip():
                missing_headers.append(header_name)
            else:
                header_values[field] = raw_value.strip()
        if missing_headers:
            raise HTTPException(status_code=401, detail="Missing Epistula authentication headers")

        body = await _get_request_body(request)

        context = cls(
            signature=signature_value,
            body=body,
            **header_values,
        )
        setattr(request.state, _EP_CONTEXT_ATTR, context)
        return context


async def _get_request_body(request: Request) -> bytes:
    """Cache and return the request body for reuse within the same request."""
    cached_body = getattr(request.state, _BODY_CACHE_ATTR, None)
    if cached_body is not None:
        return cached_body
    body = await request.body()
    setattr(request.state, _BODY_CACHE_ATTR, body)
    request._body = body  # Allow downstream consumers to reuse the cached body
    return body


def ensure_config_loaded(path: Optional[str] = None, *, reload: bool = False) -> Config:
    global CONFIG
    if CONFIG is None or reload:
        CONFIG = load_config(path)
    return CONFIG


def get_config() -> Config:
    config = CONFIG if CONFIG is not None else ensure_config_loaded()
    if config is None:
        raise RuntimeError("Configuration not loaded")
    return config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    global epistula_verifier

    logger.info("Starting Bittensor Miner Reverse Proxy")

    try:
        config = ensure_config_loaded()
        logger.info("Loaded configuration from {}", config.source_path)
        logger.info(
            "Configuration values:\n{}",
            json.dumps(config.masked_dict(), indent=2, sort_keys=True),
        )
        logger.info("Miner hotkey loaded: {}", config.auth.miner_hotkey)
    except Exception as exc:
        logger.error(f"Failed to load configuration: {exc}")
        raise

    try:
        epistula_verifier = EpistulaVerifier(
            miner_hotkey=config.auth.miner_hotkey,
            allowed_delta_ms=config.auth.allowed_delta_ms,
        )
        logger.info("Epistula verifier initialized successfully")
    except Exception as exc:
        logger.error(f"Failed to initialize Epistula verifier: {exc}")
        raise

    yield

    logger.info("Shutting down Bittensor Miner Reverse Proxy")


app = FastAPI(
    title="Bittensor Miner Reverse Proxy",
    description="Reverse proxy server for Bittensor subnet miner with Epistula authentication",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_epistula_auth(request: Request) -> bool:
    """Dependency to verify Epistula authentication."""
    if not epistula_verifier:
        raise HTTPException(status_code=500, detail="Authentication service not available")

    try:
        context = await EpistulaRequestContext.from_request(request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to prepare authentication context: {exc}")
        raise HTTPException(status_code=400, detail="Unable to process authentication headers")

    now_ms = int(time.time() * 1000)

    error = await epistula_verifier.verify_signature(
        signature=context.signature,
        body=context.body,
        timestamp=context.timestamp,
        uuid_str=context.uuid_str,
        signed_for=context.signed_for,
        signed_by=context.signed_by,
        now=now_ms,
        path=request.url.path,
    )

    if error:
        logger.warning(f"Epistula authentication failed: {error}")
        raise HTTPException(status_code=401, detail=error)

    allowed_senders = get_config().auth.allowed_senders
    if allowed_senders and context.signed_by not in allowed_senders:
        logger.warning("Epistula sender {} is not permitted", context.signed_by)
        raise HTTPException(status_code=403, detail="Sender not allowed")

    logger.debug(
        "Epistula authentication succeeded: signed_by={}, signed_for={}, uuid={}",
        context.signed_by,
        context.signed_for,
        context.uuid_str,
    )

    setattr(request.state, "epistula_signed_by", context.signed_by)

    return True


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "bittensor-miner-reverse-proxy"}


@app.get("/status")
async def status():
    """Status endpoint with service information."""
    return {
        "status": "running",
        "service": "bittensor-miner-reverse-proxy",
        "version": "0.1.0",
        "epistula_ready": epistula_verifier is not None,
    }


@app.get("/check/{identifier}")
async def check_identifier(identifier: str):
    """Return 200 if identifier matches miner hotkey, otherwise 404."""
    expected_hotkey = get_config().auth.miner_hotkey
    if identifier == expected_hotkey:
        return {"status": "ok", "identifier": identifier}
    raise HTTPException(status_code=404, detail="Identifier not recognized")


@app.get("/capacity")
async def capacity(authenticated: bool = Depends(verify_epistula_auth)):
    """Return supported capacity information."""
    return {
        "inference": ["base"],
        "training": ["H100pcie"],
    }


async def _proxy_request(request: Request, destination_base_url: str) -> Response:
    async with httpx.AsyncClient() as client:
        try:
            body = await _get_request_body(request)
            response = await client.post(
                f"{destination_base_url}{request.url.path}",
                headers=request.headers,
                content=body,
                timeout=300.0,
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
            )
        except httpx.RequestError as exc:
            logger.error(f"Error forwarding request to backend: {exc}")
            raise HTTPException(status_code=502, detail="Error connecting to backend service")


@app.post("/train")
async def proxy_training_request(
    request: Request,
    authenticated: bool = Depends(verify_epistula_auth),
):
    """Proxy training requests to the training server."""
    training_server_url = get_config().services.training_server_url
    return await _proxy_request(request, training_server_url)


@app.post("/inference")
async def proxy_inference_request(
    request: Request,
    authenticated: bool = Depends(verify_epistula_auth),
):
    """Proxy inference requests to the inference server."""
    inference_server_url = get_config().services.inference_server_url
    return await _proxy_request(request, inference_server_url)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


def main():
    """Main entry point for the server."""
    config = ensure_config_loaded()

    log_level = str(config.server.log_level).upper()
    reload_enabled = bool(config.server.reload)

    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    host = config.server.host
    port = int(config.server.port)

    logger.info("Starting server on {}:{} (reload={})", host, port, reload_enabled)

    uvicorn.run(
        "reverse_proxy.server:app",
        host=host,
        port=port,
        reload=reload_enabled,
        access_log=True,
        log_level=log_level.lower(),
    )


if __name__ == "__main__":
    main()
