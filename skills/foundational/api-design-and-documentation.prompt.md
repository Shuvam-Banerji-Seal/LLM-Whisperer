# API Design & Documentation: REST APIs, OpenAPI, and AsyncAPI

**Author**: Shuvam Banerji Seal  
**Category**: Foundational Skills  
**Difficulty**: Intermediate  
**Last Updated**: April 2026

## Problem Statement

Well-designed APIs are critical for modern systems. This skill covers:
- **REST API Principles**: RESTful design patterns
- **OpenAPI 3.1**: API specification and documentation
- **AsyncAPI**: Event-driven API specification
- **FastAPI Integration**: Automatic documentation generation
- **API Versioning**: Managing breaking changes
- **API Security**: Authentication and authorization
- **Client Generation**: Automated client code from specs

---

## Theoretical Foundations

### 1. Richardson Maturity Model

```
Level 3: HATEOAS (Hypermedia As The Engine Of Application State)
        Response includes links for next actions
        Fully RESTful

Level 2: HTTP Methods + Resources
        GET /api/users/123
        POST /api/users
        PUT /api/users/123
        DELETE /api/users/123

Level 1: Resources (avoid tunneling)
        GET /api/getUser?id=123  [Bad]
        GET /api/users/123       [Good]

Level 0: HTTP as transport (RPC)
        POST /api with action parameter
```

### 2. HTTP Status Code Matrix

```
2xx Success          3xx Redirection     4xx Client Error    5xx Server Error
200 OK               301 Moved          400 Bad Request     500 Internal Error
201 Created          302 Found          401 Unauthorized    502 Bad Gateway
202 Accepted         304 Not Modified   403 Forbidden       503 Unavailable
204 No Content       307 Temporary      404 Not Found       504 Timeout
206 Partial          308 Permanent      409 Conflict
```

### 3. API Versioning Strategy

```
Version Strategies:

1. URL Path: /api/v1/users vs /api/v2/users
2. Query: /api/users?version=2
3. Header: X-API-Version: 2
4. Content Negotiation: Accept: application/vnd.api+json;version=2

Backward Compatibility Decision Tree:
Can client ignore new fields? → Yes = Safe
Must understand new behavior? → Yes = Breaking change
```

---

## Comprehensive Code Examples

### Example 1: FastAPI with OpenAPI

```python
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid

# Initialize FastAPI app with documentation
app = FastAPI(
    title="User Management API",
    description="API for managing users with automatic OpenAPI docs",
    version="1.0.0",
    docs_url="/api/docs",          # Swagger UI
    redoc_url="/api/redoc",         # ReDoc
    openapi_url="/api/openapi.json" # OpenAPI schema
)

# Data models with documentation
class User(BaseModel):
    """User model in system."""
    
    id: str = Field(
        ...,
        title="User ID",
        description="Unique identifier for user",
        example="550e8400-e29b-41d4-a716-446655440000"
    )
    name: str = Field(
        ...,
        title="Full Name",
        min_length=1,
        max_length=100,
        example="John Doe"
    )
    email: str = Field(
        ...,
        title="Email",
        regex=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        example="john@example.com"
    )
    age: Optional[int] = Field(
        None,
        ge=0,
        le=150,
        title="Age",
        description="User age (0-150)"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        title="Created At",
        description="Timestamp when user was created"
    )


class UserCreate(BaseModel):
    """Request model for creating user (no id, timestamps)."""
    
    name: str = Field(..., min_length=1, max_length=100)
    email: str
    age: Optional[int] = Field(None, ge=0, le=150)


class ErrorResponse(BaseModel):
    """Standard error response format."""
    
    error: str = Field(..., title="Error Type")
    message: str = Field(..., title="Error Message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


# In-memory storage for example
users_db: dict[str, User] = {}


# API Endpoints with documentation
@app.get(
    "/api/v1/users",
    response_model=List[User],
    summary="List all users",
    description="Retrieve paginated list of users",
    tags=["Users"]
)
async def list_users(
    skip: int = Query(0, ge=0, description="Number of users to skip"),
    limit: int = Query(10, ge=1, le=100, description="Max users to return"),
) -> List[User]:
    """
    Get list of users with pagination.
    
    **Query Parameters:**
    - `skip`: Number of records to skip (default: 0)
    - `limit`: Maximum records to return (default: 10, max: 100)
    
    **Response:**
    - Returns array of User objects
    
    **Examples:**
    - `/api/v1/users` - First 10 users
    - `/api/v1/users?skip=10&limit=5` - Users 11-15
    """
    users = list(users_db.values())
    return users[skip : skip + limit]


@app.get(
    "/api/v1/users/{user_id}",
    response_model=User,
    responses={
        200: {"description": "User found"},
        404: {"description": "User not found", "model": ErrorResponse}
    },
    summary="Get user by ID",
    tags=["Users"]
)
async def get_user(user_id: str) -> User:
    """
    Retrieve specific user by ID.
    
    **Path Parameters:**
    - `user_id`: UUID of the user
    
    **Responses:**
    - 200: User found
    - 404: User not found
    """
    if user_id not in users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
            headers={"X-Error-Code": "USER_NOT_FOUND"}
        )
    
    return users_db[user_id]


@app.post(
    "/api/v1/users",
    response_model=User,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "User created"},
        400: {"description": "Invalid input", "model": ErrorResponse}
    },
    summary="Create new user",
    tags=["Users"]
)
async def create_user(user_create: UserCreate) -> User:
    """
    Create new user in system.
    
    **Request Body:**
    - `name`: User's full name (required)
    - `email`: User's email address (required)
    - `age`: User's age (optional)
    
    **Response:**
    - 201: User successfully created
    - 400: Invalid input data
    """
    # Validate email uniqueness
    if any(u.email == user_create.email for u in users_db.values()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    user_id = str(uuid.uuid4())
    user = User(
        id=user_id,
        name=user_create.name,
        email=user_create.email,
        age=user_create.age
    )
    
    users_db[user_id] = user
    return user


@app.put(
    "/api/v1/users/{user_id}",
    response_model=User,
    summary="Update user",
    tags=["Users"]
)
async def update_user(user_id: str, user_update: UserCreate) -> User:
    """
    Update existing user.
    
    **Path Parameters:**
    - `user_id`: UUID of user to update
    
    **Request Body:**
    - Fields to update (partial update supported)
    """
    if user_id not in users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user = users_db[user_id]
    user.name = user_update.name
    user.email = user_update.email
    if user_update.age is not None:
        user.age = user_update.age
    
    return user


@app.delete(
    "/api/v1/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete user",
    tags=["Users"]
)
async def delete_user(user_id: str) -> None:
    """
    Delete user from system.
    
    **Path Parameters:**
    - `user_id`: UUID of user to delete
    
    **Response:**
    - 204: User successfully deleted
    - 404: User not found
    """
    if user_id not in users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    del users_db[user_id]


# Health check endpoint
@app.get(
    "/api/health",
    summary="Health check",
    tags=["System"]
)
async def health_check():
    """
    System health check endpoint.
    
    Returns:
        - status: "healthy" or "unhealthy"
        - timestamp: Current server time
    """
    return {"status": "healthy", "timestamp": datetime.utcnow()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # Access docs:
    # - Swagger UI: http://localhost:8000/api/docs
    # - ReDoc: http://localhost:8000/api/redoc
    # - OpenAPI JSON: http://localhost:8000/api/openapi.json
```

### Example 2: AsyncAPI for Event-Driven APIs

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum
import json

# Event models for AsyncAPI
class EventType(str, Enum):
    """Event types in system."""
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"


class UserEvent(BaseModel):
    """Event published when user action occurs."""
    
    event_type: EventType = Field(..., title="Event Type")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: str = Field(..., title="User ID")
    user_data: dict = Field(..., title="User Data")
    correlation_id: Optional[str] = Field(
        None,
        title="Correlation ID",
        description="For distributed tracing"
    )


# AsyncAPI specification as YAML (in practice)
ASYNCAPI_SPEC = """
asyncapi: 3.0.0
info:
  title: User Events API
  version: 1.0.0
  description: Real-time user events
  contact:
    name: API Support
    url: https://example.com/support

channels:
  user.events:
    address: user-topic
    messages:
      UserCreated:
        payload:
          type: object
          properties:
            event_type:
              type: string
              enum:
                - user.created
                - user.updated
                - user.deleted
            timestamp:
              type: string
              format: date-time
            user_id:
              type: string
              format: uuid
            user_data:
              type: object
            correlation_id:
              type: string
              format: uuid

operations:
  publishUserEvent:
    action: send
    channel:
      $ref: '#/channels/user.events'
    summary: Publish user event
    messages:
      - $ref: '#/channels/user.events/messages/UserCreated'
  
  subscribeToUserEvents:
    action: receive
    channel:
      $ref: '#/channels/user.events'
    summary: Subscribe to user events
    messages:
      - $ref: '#/channels/user.events/messages/UserCreated'
"""


# WebSocket example for real-time events
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


@app.websocket("/api/v1/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time events.
    
    Clients connect and receive UserEvent messages in real-time.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Receive and broadcast events
            data = await websocket.receive_text()
            event = UserEvent.parse_raw(data)
            await manager.broadcast(event.dict())
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### Example 3: API Versioning and Backward Compatibility

```python
from fastapi import APIRouter, Header, status
from typing import Optional

# V1 Router
v1_router = APIRouter(prefix="/api/v1", tags=["v1"])

@v1_router.get("/users/{user_id}")
async def get_user_v1(user_id: str):
    """V1: Returns user with basic fields."""
    return {
        "id": user_id,
        "name": "John Doe",
        "email": "john@example.com"
    }


# V2 Router with additional fields
v2_router = APIRouter(prefix="/api/v2", tags=["v2"])

@v2_router.get("/users/{user_id}")
async def get_user_v2(user_id: str):
    """V2: Returns user with additional metadata."""
    return {
        "id": user_id,
        "name": "John Doe",
        "email": "john@example.com",
        "created_at": "2024-01-01T00:00:00Z",
        "last_login": "2024-04-06T10:30:00Z",
        "status": "active"
    }


# Header-based versioning
@app.get("/api/users/{user_id}")
async def get_user_versioned(
    user_id: str,
    x_api_version: Optional[str] = Header("1")
):
    """
    Support multiple API versions via header.
    
    Usage:
        curl -H "X-API-Version: 2" /api/users/123
    """
    if x_api_version == "2":
        return await get_user_v2(user_id)
    else:
        return await get_user_v1(user_id)


# Deprecated endpoint with warning
@app.get(
    "/api/v1/users/search",
    deprecated=True,
    summary="Search users (deprecated)"
)
async def search_users_deprecated(q: str):
    """
    **DEPRECATED**: Use POST /api/v2/users/search instead.
    
    This endpoint will be removed on 2026-12-31.
    """
    return {
        "warning": "This endpoint is deprecated",
        "message": "Use POST /api/v2/users/search instead"
    }


app.include_router(v1_router)
app.include_router(v2_router)
```

### Example 4: API Security with Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials
from jwt import PyJWT, ExpiredSignatureError
from datetime import datetime, timedelta
import os

# Security scheme
security = HTTPBearer()

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-key")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = timedelta(hours=24)


def create_token(user_id: str) -> str:
    """Create JWT token for user."""
    import jwt
    
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + JWT_EXPIRATION,
        "iat": datetime.utcnow()
    }
    
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(credentials: HTTPAuthCredentials = Depends(security)) -> str:
    """Verify JWT token and return user_id."""
    import jwt
    
    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM]
        )
        user_id: str = payload.get("user_id")
        
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return user_id
    
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/api/v1/auth/login")
async def login(username: str, password: str):
    """
    Authenticate user and return JWT token.
    
    **Request Body:**
    - `username`: User's username
    - `password`: User's password
    
    **Response:**
    - `access_token`: JWT token for authenticated requests
    """
    # Validate credentials (simplified)
    if username == "demo" and password == "password":
        token = create_token(user_id="user-123")
        return {
            "access_token": token,
            "token_type": "bearer"
        }
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials"
    )


@app.get(
    "/api/v1/users/me",
    summary="Get current user profile"
)
async def get_current_user(user_id: str = Depends(verify_token)):
    """
    Get current user's profile.
    
    **Security**: Requires JWT Bearer token in Authorization header
    
    **Usage:**
    ```
    curl -H "Authorization: Bearer <token>" /api/v1/users/me
    ```
    """
    return {
        "user_id": user_id,
        "name": "Current User",
        "email": "user@example.com"
    }
```

### Example 5: OpenAPI Client Generation

```python
from typing import Optional
import subprocess

class OpenAPICodegen:
    """Generate client code from OpenAPI specification."""
    
    def __init__(self, openapi_url: str = "http://localhost:8000/api/openapi.json"):
        self.openapi_url = openapi_url
    
    def generate_python_client(self, output_dir: str = "./client"):
        """
        Generate Python client from OpenAPI spec.
        
        Requires: pip install openapi-generator-cli
        """
        try:
            subprocess.run([
                "openapi-generator-cli", "generate",
                "-i", self.openapi_url,
                "-g", "python",
                "-o", output_dir,
                "-c", "config.yaml"
            ], check=True)
            
            print(f"✓ Generated Python client in {output_dir}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Code generation failed: {e}")
    
    def generate_typescript_client(self, output_dir: str = "./client"):
        """Generate TypeScript client."""
        try:
            subprocess.run([
                "openapi-generator-cli", "generate",
                "-i", self.openapi_url,
                "-g", "typescript",
                "-o", output_dir
            ], check=True)
            
            print(f"✓ Generated TypeScript client in {output_dir}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Generation failed: {e}")


# Usage
"""
Install code generator:
pip install openapi-generator-cli

Generate Python client:
openapi-generator-cli generate -i http://localhost:8000/openapi.json -g python -o ./client

Generated client usage:
from client.apis.users_api import UsersAPI
api = UsersAPI()
user = api.get_user(user_id="123")
"""
```

---

## Step-by-Step Implementation Guide

### 1. Creating FastAPI with Documentation

**Step 1.1: Install FastAPI**
```bash
pip install fastapi uvicorn pydantic
```

**Step 1.2: Create API with models**
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="My API")

class Item(BaseModel):
    name: str
    description: str = None

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    return {"item_id": item_id}
```

**Step 1.3: Run server**
```bash
uvicorn main:app --reload
# Access docs at http://localhost:8000/docs
```

### 2. Securing APIs

**Step 2.1: Install JWT library**
```bash
pip install python-jose cryptography
```

**Step 2.2: Add authentication**
```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.get("/protected")
async def protected(credentials: HTTPAuthCredentials = Depends(security)):
    return {"message": "Authenticated"}
```

### 3. API Versioning

**Step 3.1: Create version routers**
```python
from fastapi import APIRouter

v1_router = APIRouter(prefix="/api/v1")
v2_router = APIRouter(prefix="/api/v2")

app.include_router(v1_router)
app.include_router(v2_router)
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Leaky Implementation Details in API
**Problem**: API exposes internal database structure
```python
@app.get("/users")
def get_users():
    return db.query(User).all()  # Exposes ORM details
```

**Solution**: Use response models
```python
class UserResponse(BaseModel):
    id: str
    name: str

@app.get("/users", response_model=List[UserResponse])
def get_users():
    return users
```

### Pitfall 2: Breaking Changes Without Versioning
**Problem**: Removing field breaks client code
```python
# V1 had: {"user_id": 123, "name": "John"}
# V2 changes: {"id": 123, "full_name": "John"}
```

**Solution**: Introduce new version, keep old version
```python
# Keep /api/v1/users returning old format
# Add /api/v2/users with new format
```

### Pitfall 3: Insufficient Error Documentation
**Problem**: Clients don't know what errors to expect
```python
@app.get("/users/{user_id}")
def get_user(user_id: int):
    # What if user not found? No documentation
    pass
```

**Solution**: Document all responses
```python
@app.get("/users/{user_id}", responses={
    200: {"description": "User found"},
    404: {"description": "User not found"}
})
def get_user(user_id: int):
    pass
```

---

## Authoritative Sources

1. **OpenAPI 3.1 Specification**: https://spec.openapis.org/oas/v3.1.0
2. **AsyncAPI 3.0**: https://www.asyncapi.com/
3. **FastAPI Documentation**: https://fastapi.tiangolo.com/
4. **REST API Best Practices**: https://restfulapi.net/
5. **RFC 7231 - HTTP Methods**: https://tools.ietf.org/html/rfc7231
6. **RFC 7232 - HTTP Semantics**: https://tools.ietf.org/html/rfc7232
7. **Semantic Versioning**: https://semver.org/
8. **API Security Guidelines**: https://owasp.org/www-project-api-security/
9. **RESTful Design by Leonard Richardson**: https://www.oreilly.com/library/view/restful-web-services/9780596529260/
10. **Building Web APIs by Mike Amundsen**: https://www.oreilly.com/library/view/building-web-apis/9781492053131/

---

## Summary

Design robust APIs through:
- RESTful principles following Richardson maturity model
- Comprehensive OpenAPI documentation
- Semantic versioning for backward compatibility
- Security with authentication and authorization
- Async patterns for real-time communication
- Automated client generation

These patterns enable production-grade APIs for LLM systems, microservices, and distributed applications.
