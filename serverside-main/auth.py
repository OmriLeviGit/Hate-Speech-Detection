from fastapi import HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt, JWTError

from credentials import JWT_SECRET_KEY



def login_required(func):
    async def wrapper(credentials: HTTPAuthorizationCredentials = Depends(auth), *args, **kwargs):
        try:
            # Verify and decode the token
            payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=["HS256"])
            key = payload.get("key")
            if key:
                # Process the request with the authenticated key
                return await func(passcode=key, *args, **kwargs)
            else:
                raise HTTPException(status_code=401, detail="Invalid token")
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__

    params = list(inspect.signature(func).parameters.values()) + list(inspect.signature(wrapper).parameters.values())
    wrapper.__signature__ = inspect.signature(func).replace(
        parameters=[
            # Use all parameters from handler
            *filter(lambda p: p.name != 'passcode', inspect.signature(func).parameters.values()),

            # Skip *args and **kwargs from wrapper parameters:
            *filter(
                lambda p: p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD),
                inspect.signature(wrapper).parameters.values()
            )
        ]
    )

    return wrapper


def generate_token(key: str):
    # Set the token expiration time
    expire = datetime.utcnow() + timedelta(hours=3)
    # Create the payload containing the key
    payload = {"key": key, "exp": expire}
    # Generate the JWT token
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")
    return token