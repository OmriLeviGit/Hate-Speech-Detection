import inspect
import os
from datetime import datetime, timedelta
from fastapi.security import HTTPBearer

from fastapi import HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt, JWTError
from dotenv import load_dotenv

load_dotenv('.local.env')
JWT_SECRET_KEY = os.environ.get('JWT_TOKEN')


auth = HTTPBearer()

def login_required(func):
    async def wrapper(credentials: HTTPAuthorizationCredentials = Depends(auth), *args, **kwargs):
        try:

            # print(f"Received Token: {credentials.credentials}")  # DEBUG: Print the token
            
            payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=["HS256"])

            user_id = payload.get("user_id")  # Extract user_id from token

            # print(f"Decoded Payload: {payload}")  # DEBUG: Print payload content

            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid token")
        
            return await func(user_id=user_id, *args, **kwargs)  # Pass user_id to the endpoint

        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__

    params = list(inspect.signature(func).parameters.values()) + list(inspect.signature(wrapper).parameters.values()) # not used?
    wrapper.__signature__ = inspect.signature(func).replace(
        parameters=[
            *filter(lambda p: p.name != 'user_id', inspect.signature(func).parameters.values()),

            *filter(
                lambda p: p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD),
                inspect.signature(wrapper).parameters.values()
            )
        ]
    )

    return wrapper


def generate_token(user_id: str):
    # Set the token expiration time
    expire = datetime.utcnow() + timedelta(hours=3)
    # Create the payload containing the key
    payload = {"user_id": user_id, "exp": expire}
    # Generate the JWT token
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")
    return token