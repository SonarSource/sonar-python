from fastapi import FastAPI, APIRouter

app = FastAPI()
router = APIRouter()

@app.get("/items/{item_id}")
def noncompliant_missing_path_param():  # Noncompliant {{Add path parameter "item_id" to the function signature.}}
    return {"message": "Hello"}

@app.get("/users/{user_id}/items/{item_id}")
def noncompliant_missing_one_of_multiple_params(user_id: int):  # Noncompliant {{Add path parameter "item_id" to the function signature.}}
    return {"user_id": user_id}

@app.get("/items/{item_id}")
def noncompliant_positional_only_param(item_id: int, /):  # Noncompliant {{Path parameter "item_id" should not be positional-only.}}
    return {"item_id": item_id}

@app.get("/things/{thing_id}")
async def noncompliant_async_missing_param(query: str):  # Noncompliant 
    return {"query": query}

@app.get("/users/{user_id}/posts/{post_id}")
def noncompliant_multiple_positional_only(user_id: int, post_id: int, /):  # Noncompliant 2
    return {"user_id": user_id}

@router.put("/items/{item_id}")
def noncompliant_router_missing_param():  # Noncompliant 
    return {"updated": True}

@app.post("/items/{item_id}")
def noncompliant_post_missing():  # Noncompliant 
    pass

@app.put("/items/{item_id}")
def noncompliant_put_missing():  # Noncompliant 
    pass

@app.delete("/items/{item_id}")
def noncompliant_delete_missing():  # Noncompliant 
    pass

@app.patch("/items/{item_id}")
def noncompliant_patch_missing():  # Noncompliant 
    pass

@app.options("/items/{item_id}")
def noncompliant_options_missing():  # Noncompliant 
    pass

@app.head("/items/{item_id}")
def noncompliant_head_missing():  # Noncompliant 
    pass

@app.trace("/items/{item_id}")
def noncompliant_trace_missing():  # Noncompliant 
    pass

@app.get("/items/{item_id:int}")
def noncompliant_with_converter_missing():  # Noncompliant 
    pass

@app.get("/users/{user_id}/items/{item_id}")
def noncompliant_mixed_positional_only_and_missing(user_id: int, /):  # Noncompliant 2
    pass

@app.get("/a/{x}/b/{y}/c/{z}")
def noncompliant_all_params_missing():  # Noncompliant 3
    pass

@app.get(path="/items/{item_id}")
def noncompliant_path_keyword_missing():  # Noncompliant 
    return {}

@app.get("/items/{item_id}")
def compliant_basic(item_id: int):
    return {"item_id": item_id}

@app.get("/users/{user_id}/items/{item_id}")
def compliant_multiple_params(user_id: int, item_id: int):
    return {"user_id": user_id, "item_id": item_id}

@app.get("/users/{user_id}/items/{item_id}")
def compliant_reordered_params(item_id: int, user_id: int):
    return {"user_id": user_id, "item_id": item_id}

@app.get("/things/{thing_id}")
async def compliant_async_with_query(thing_id: int, query: str):
    return {"thing_id": thing_id, "query": query}

@app.get("/items")
def compliant_static_route():
    return {"items": []}

@router.put("/items/{item_id}")
def compliant_router_with_param(item_id: int):
    return {"updated": True}

@app.get("/items/{item_id}")
def compliant_keyword_only(*, item_id: int):
    return {"item_id": item_id}

@app.post("/items/{item_id}")
def compliant_post(item_id: int):
    pass

@app.put("/items/{item_id}")
def compliant_put(item_id: int):
    pass

@app.delete("/items/{item_id}")
def compliant_delete(item_id: int):
    pass

@app.get("/items/{item_id}")
def compliant_with_default(item_id: int = 1):
    return {"item_id": item_id}

@app.get("/items/{item_id:int}")
def compliant_with_converter(item_id: int):
    return {"item_id": item_id}

@app.get("/items/{item_id}")
def compliant_with_kwargs(**kwargs):
    return kwargs

@app.get("/items/{item_id}")
def compliant_with_args_kwargs(*args, **kwargs):
    return {"args": args, "kwargs": kwargs}

@app.get(status_code=200, path="/items/{item_id}")
def compliant_path_keyword(item_id: int):
    return {"item_id": item_id}

# --- Edge cases ---

def some_other_decorator(path):
    def wrapper(func):
        return func
    return wrapper

@some_other_decorator("/items/{item_id}")
def compliant_not_fastapi_decorator():
    pass

@app.get("/static")
def compliant_no_path_params():
    pass

@app.get("/items")
def compliant_extra_query_params(query: str):
    return {"query": query}

path = "/items/{item_id}"
@app.get(path)
def noncompliant_dynamic_path(): # Noncompliant
    pass

@app.get
def compliant_decorator_without_call():
    pass

@app.get("")
def compliant_empty_path():
    pass


@app.get(True)
def compliant_path_is_not_a_string():
    pass
