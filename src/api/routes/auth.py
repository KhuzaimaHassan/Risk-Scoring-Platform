"""Admin authentication with bounded input and per-client throttling."""
import time
from collections import OrderedDict
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm
from starlette.concurrency import run_in_threadpool
from config.settings import get_settings
from src.api.auth import ADMIN_PASSWORD_HASH, create_access_token, verify_password

router = APIRouter(tags=["Authentication"])
settings = get_settings()
_attempts = OrderedDict()

@router.post("/token", summary="Login to get access token")
async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    if ADMIN_PASSWORD_HASH is None:
        raise HTTPException(503, "Admin sign-in is not configured.")
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    failures = [t for t in _attempts.get(client, []) if now - t < 900]
    if len(failures) >= 10:
        raise HTTPException(429, "Too many sign-in attempts. Try again in 15 minutes.", headers={"Retry-After": "900"})
    _attempts[client] = failures + [now]
    _attempts.move_to_end(client)
    while len(_attempts) > 4096:
        _attempts.popitem(last=False)
    password_ok = await run_in_threadpool(verify_password, form_data.password, ADMIN_PASSWORD_HASH)
    if form_data.username != settings.admin_username or not password_ok:
        raise HTTPException(401, "Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    _attempts.pop(client, None)
    token = create_access_token({"sub": settings.admin_username}, timedelta(minutes=settings.access_token_expire_minutes))
    return {"access_token": token, "token_type": "bearer"}
