"""
src/api/auth.py
----------------
Authentication and JWT utilities.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
import hashlib
import hmac
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError

from config.settings import get_settings

settings = get_settings()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.api_prefix}/token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    encoded = plain_password.encode('utf-8')
    return bool(encoded) and len(encoded) <= 72 and bcrypt.checkpw(encoded, hashed_password.encode('utf-8'))

def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Generate the hash once on startup for the configured admin user
ADMIN_PASSWORD_HASH = get_password_hash(settings.admin_password.get_secret_value()) if settings.admin_password.get_secret_value() else None

def signing_key():
    # Password rotation invalidates previously issued admin tokens, including default-login tokens.
    return hmac.new(settings.secret_key.get_secret_value().encode(),
                    settings.admin_password.get_secret_value().encode(), hashlib.sha256).digest()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        signing_key(),
        algorithm=settings.jwt_algorithm
    )
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            signing_key(),
            algorithms=[settings.jwt_algorithm]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except InvalidTokenError:
        raise credentials_exception
    
    if username != settings.admin_username:
        raise credentials_exception
        
    return username
