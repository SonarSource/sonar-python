from typing import List, Optional, Union
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel

app = FastAPI()
router = APIRouter()

class Item(BaseModel):
    name: str

class User(BaseModel):
    id: int

class UserPublic(BaseModel):
    id: int

class UserInternal(BaseModel):
    id: int
    password: str

@app.post("/items/", response_model=Item)  # Noncompliant {{Remove this redundant "response_model" parameter; it duplicates the return type annotation.}}
#                    ^^^^^^^^^^^^^^^^^^^
async def create_item(item: Item) -> Item:
    return item

@app.get("/items/{item_id}", response_model=Item)  # Noncompliant
def get_item(item_id: int) -> Item:
    return fetch_item(item_id)

@app.put("/items/{item_id}", response_model=Item)  # Noncompliant
def update_item(item_id: int, item: Item) -> Item:
    return item

@app.delete("/items/{item_id}", response_model=Item)  # Noncompliant
def delete_item(item_id: int) -> Item:
    return removed_item

@app.patch("/items/{item_id}", response_model=Item)  # Noncompliant
def patch_item(item_id: int) -> Item:
    return patched_item

@app.options("/items/", response_model=Item)  # Noncompliant
def options_item() -> Item:
    return item

@app.head("/items/", response_model=Item)  # Noncompliant
def head_item() -> Item:
    return item

@app.trace("/items/", response_model=Item)  # Noncompliant
def trace_item() -> Item:
    return item

@router.post("/items/", response_model=Item)  # Noncompliant
def create_item_router(item: Item) -> Item:
    return item

@app.get("/items/", response_model=List[Item])  # Noncompliant
def get_items() -> List[Item]:
    return items

@app.get("/item/{item_id}", response_model=Optional[Item])  # Noncompliant
def get_optional_item(item_id: int) -> Optional[Item]:
    return maybe_item

@app.post("/items/", response_model=Item, status_code=201)  # Noncompliant
def create_item_first(item: Item) -> Item:
    return item

@app.post("/items/", status_code=201, response_model=Item, tags=["items"])  # Noncompliant
def create_item_middle(item: Item) -> Item:
    return item

@app.post("/items/", status_code=201, tags=["items"], response_model=Item)  # Noncompliant
def create_item_last(item: Item) -> Item:
    return item

@app.post("/items/", response_model=Item)  # Noncompliant
def create_item_sync(item: Item) -> Item:
    return item

@app.post("/items/")
async def create_item_no_model(item: Item) -> Item:
    return item

@app.post("/items/", response_model=Item)
def create_item_no_return_type_hint(item: Item):
    return item

@app.get("/users/{user_id}", response_model=UserPublic)
def compliant_overriding_response_model(user_id: int) -> UserInternal:
    return fetch_user_with_password(user_id)

@app.get("/items/", response_model=List[UserPublic])
def fastapi_with_generic_list() -> List[User]:
    return items


@some_other_decorator("/items/", response_model=Item)
def non_fastapi_endpoint(item: Item) -> Item:
    return item

def not_an_endpoint(item: Item) -> Item:
    return item

@app.post
def invalid_decorator() -> Item:
    return item

@app.post("/items/")
def no_types(item):
    return item
