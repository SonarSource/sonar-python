from typing import Annotated
from fastapi import Depends, FastAPI, Query, Path, Body, Header, Cookie, Form, File
from fastapi import APIRouter

app = FastAPI()
router = APIRouter()

def get_db():
    return "database_connection"

def compliant_depends_in_annotated():
    @app.get("/items/")
    def read_items(db: Annotated[str, Depends(get_db)]):
        return {"db": db}

def compliant_query_in_annotated():
    @app.get("/search/")
    def search_items(q: Annotated[str | None, Query(max_length=50)] = None):
        return {"query": q}

def compliant_path_in_annotated():
    @app.get("/items/{item_id}")
    def read_item(item_id: Annotated[int, Path(gt=0)]):
        return {"item_id": item_id}

def compliant_body_in_annotated():
    @app.post("/items/")
    def create_item(item: Annotated[dict, Body()]):
        return item

def compliant_header_in_annotated():
    @app.get("/items/")
    def read_items(x_token: Annotated[str | None, Header()] = None):
        return {"X-Token": x_token}

def compliant_cookie_in_annotated():
    @app.get("/items/")
    def read_items(session: Annotated[str | None, Cookie()] = None):
        return {"session": session}

def compliant_regular_parameter():
    @app.get("/items/")
    def read_items(skip: int = 0, limit: int = 10):
        return {"skip": skip, "limit": limit}

def compliant_api_router():
    @router.get("/items/")
    def read_items(db: Annotated[str, Depends(get_db)]):
        return {"db": db}

def not_a_route_handler(db = Depends(get_db)):
    return {"db": db}

def compliant_multiple_annotated():
    @app.get("/users/")
    def read_users(
        db: Annotated[str, Depends(get_db)],
        skip: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 10
    ):
        return {"db": db, "skip": skip, "limit": limit}

def noncompliant_depends_as_default():
    @app.get("/items/")
    def read_items(db = Depends(get_db)):  # Noncompliant {{Use "Annotated" type hints for FastAPI dependency injection}}
#                  ^^^^^^^^^^^^^^^^^^^^
        return {"db": db}

def noncompliant_query_as_default():
    @app.get("/search/")
    def search_items(q: str = Query(None, max_length=50)):  # Noncompliant
        return {"query": q}

def noncompliant_path_as_default():
    @app.get("/items/{item_id}")
    def read_item(item_id: int = Path(gt=0)):  # Noncompliant
        return {"item_id": item_id}

def noncompliant_body_as_default():
    @app.post("/items/")
    def create_item(item: dict = Body()):  # Noncompliant
        return item

def noncompliant_header_as_default():
    @app.get("/items/")
    def read_items(x_token: str = Header()):  # Noncompliant
        return {"X-Token": x_token}

def noncompliant_cookie_as_default():
    @app.get("/items/")
    def read_items(session: str = Cookie()):  # Noncompliant
        return {"session": session}

def noncompliant_multiple_old_syntax():
    @app.get("/users/")
    def read_users(
        db = Depends(get_db),  # Noncompliant
        skip: int = Query(0, ge=0),  # Noncompliant
        limit: int = Query(10, ge=1, le=100)  # Noncompliant
    ):
        return {"db": db, "skip": skip, "limit": limit}

def noncompliant_api_router():
    @router.get("/items/")
    def read_items(db = Depends(get_db)):  # Noncompliant
        return {"db": db}

def noncompliant_async_function():
    @app.get("/items/")
    async def read_items(db = Depends(get_db)):  # Noncompliant
        return {"db": db}

def noncompliant_complex_dependency():
    async def common_parameters(q: str | None = None, skip: int = 0):
        return {"q": q, "skip": skip}

    @app.get("/items/")
    async def read_items(commons: dict = Depends(common_parameters)):  # Noncompliant
        return commons

def noncompliant_all_http_methods():
    @app.post("/items/")
    def create(db = Depends(get_db)):  # Noncompliant
        pass

    @app.put("/items/")
    def update(db = Depends(get_db)):  # Noncompliant
        pass

    @app.delete("/items/")
    def delete(db = Depends(get_db)):  # Noncompliant
        pass

    @app.patch("/items/")
    def patch(db = Depends(get_db)):  # Noncompliant
        pass

    @app.options("/items/")
    def options(db = Depends(get_db)):  # Noncompliant
        pass

    @app.head("/items/")
    def head(db = Depends(get_db)):  # Noncompliant
        pass

    @app.trace("/items/")
    def trace(db = Depends(get_db)):  # Noncompliant
        pass

def noncompliant_mixed():
    @app.get("/items/")
    def read_items(
        db = Depends(get_db),  # Noncompliant
        skip: Annotated[int, Query(ge=0)] = 0
    ):
        return {"db": db, "skip": skip}

def noncompliant_form_parameter():
    @app.post("/form/")
    def form(name: str = Form(...)):  # Noncompliant
        pass

def noncompliant_file_parameter():
    @app.post("/upload/")
    def upload(file = File(...)):  # Noncompliant
        pass

def edge_case_annotated_for_other_purpose():
    @app.get("/items/")
    def read_items(value: Annotated[int, "some metadata"] = 0):
        return {"value": value}

def edge_case_nested_annotated():
    from typing import List
    @app.get("/items/")
    def read_items(items: Annotated[list[int], Query()] = []):
        return {"items": items}

def edge_case_not_call_expression():
    some_value = Query(None)
    @app.get("/items/")
    def read_items(q = some_value):
        return {"q": q}

def edge_case_decorator_without_call():
    def my_decorator(func):
        return func

    @my_decorator
    def read_items(q = Query(None)):
        return {"q": q}

def edge_case_function_without_parameters():
    @app.get("/items/")
    def read_items():
        return {"message": "no parameters"}

def edge_case_non_fastapi_call_default():
    def some_function():
        return "value"

    @app.get("/items/")
    def read_items(value = some_function()):
        return {"value": value}

def compliant_form_in_annotated():
    @app.post("/form/")
    def form(name: Annotated[str, Form(...)]):
        pass

def compliant_file_in_annotated():
    @app.post("/upload/")
    def upload(file: Annotated[bytes, File(...)]):
        pass

def edge_case_annotated_without_dependency_call():
    from typing import List
    @app.get("/items/")
    def read_items(items: Annotated[list[int], "metadata", 42] = Query()):  # Noncompliant
        return {"items": items}

def edge_case_annotated_with_non_call_expression():
    dependency_instance = Depends(get_db)
    @app.get("/items/")
    def read_items(db: Annotated[str, dependency_instance] = Depends(get_db)):  # Noncompliant
        return {"db": db}

def edge_case_subscription_not_annotated():
    from typing import List
    @app.get("/items/")
    def read_items(items: List[int] = Query()):  # Noncompliant
        return {"items": items}

def false_positive_subscription_with_qualified_name():
    import typing
    @app.get("/items/")
    def read_items(db: typing.Annotated[str, Depends(get_db)] = Depends(get_db)):  # Noncompliant
        return {"db": db}

def false_negative_annotated_with_dependency_detected():
    @app.get("/items/")
    def read_items(db: Annotated[str, Depends(get_db)] = Depends(get_db)): # Compliant
        return {"db": db}
