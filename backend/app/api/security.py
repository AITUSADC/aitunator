"""Rate limiting, trusted proxy client IP extraction, and per-session concurrency locks."""
from __future__ import annotations

import ipaddress
import os
import time
from collections import deque
from fastapi import Request


def _parse_trusted_proxies(raw: str | None) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    if not raw:
        return ()
    networks = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        try:
            if "/" in token:
                networks.append(ipaddress.ip_network(token, strict=False))
            else:
                addr = ipaddress.ip_address(token)
                prefix = 32 if addr.version == 4 else 128
                networks.append(ipaddress.ip_network(f"{addr}/{prefix}", strict=False))
        except ValueError:
            continue
    return tuple(networks)


def is_trusted_proxy(host: str | None, trusted_raw: str | None = None) -> bool:
    """Return True only if `host` belongs to the configured TRUSTED_PROXY_IPS list."""
    if not host:
        return False
    raw = trusted_raw if trusted_raw is not None else os.getenv("TRUSTED_PROXY_IPS", "")
    networks = _parse_trusted_proxies(raw)
    if not networks:
        return False
    try:
        addr = ipaddress.ip_address(host.strip())
    except ValueError:
        return False
    return any(addr in net for net in networks)


def get_client_ip(request: Request, trusted_raw: str | None = None) -> str:
    """Extract client IP, trusting X-Forwarded-For / X-Real-IP ONLY from a trusted proxy."""
    peer_host = request.client.host if request.client and request.client.host else "127.0.0.1"
    if is_trusted_proxy(peer_host, trusted_raw=trusted_raw):
        xff = request.headers.get("x-forwarded-for")
        if xff:
            first_ip = xff.split(",")[0].strip()
            if first_ip:
                return first_ip
        x_real_ip = request.headers.get("x-real-ip")
        if x_real_ip and x_real_ip.strip():
            return x_real_ip.strip()
    return peer_host


class SlidingWindowRateLimiter:
    """In-memory sliding window rate limiter (default: 10 requests / 60 seconds per IP)."""

    def __init__(self, max_requests: int = 10, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str, now: float | None = None) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        if self.max_requests <= 0:
            return True, 0
        current = now if now is not None else time.monotonic()
        cutoff = current - self.window_seconds
        bucket = self._hits.get(key)
        if bucket is None:
            bucket = deque()
            self._hits[key] = bucket
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= self.max_requests:
            retry_after = max(1, int(self.window_seconds - (current - bucket[0])) + 1)
            return False, retry_after
        bucket.append(current)
        return True, 0

    def reset(self) -> None:
        self._hits.clear()


class SessionLockRegistry:
    """Enforces at most one active request per session_id while allowing parallel sessions."""

    def __init__(self) -> None:
        self._active_sessions: set[str] = set()

    def try_acquire(self, session_id: str) -> bool:
        if session_id in self._active_sessions:
            return False
        self._active_sessions.add(session_id)
        return True

    def release(self, session_id: str) -> None:
        self._active_sessions.discard(session_id)

    @property
    def active_count(self) -> int:
        return len(self._active_sessions)
