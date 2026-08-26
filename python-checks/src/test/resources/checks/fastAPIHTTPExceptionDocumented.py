from fastapi import FastAPI, HTTPException, APIRouter

app = FastAPI()
router = APIRouter()


# Non-compliant cases

@app.get("/users/{user_id}")
def undocumented_404_exception(user_id: int):
    if ...:
        raise HTTPException(status_code=404, detail="User not found")  # Noncompliant {{Document this HTTPException with status code 404 in the "responses" parameter.}}
#             ^^^^^^^^^^^^^


@app.post("/items")
def multiple_same_status_code_undocumented(item: dict):
    if ...:
        raise HTTPException(status_code=422, detail="Name is required")  # Noncompliant {{Document this HTTPException with status code 422 in the "responses" parameter.}}
#             ^^^^^^^^^^^^^
    if ...:
        raise HTTPException(status_code=422, detail="Name too long")  # Noncompliant {{Document this HTTPException with status code 422 in the "responses" parameter.}}
#             ^^^^^^^^^^^^^


@app.put("/items/{item_id}")
def multiple_different_status_codes_undocumented(item_id: int, item: dict):
    if ...:
        raise HTTPException(status_code=404, detail="Item not found")  # Noncompliant {{Document this HTTPException with status code 404 in the "responses" parameter.}}
#             ^^^^^^^^^^^^^
    if ...:
        raise HTTPException(status_code=403, detail="Forbidden")  # Noncompliant {{Document this HTTPException with status code 403 in the "responses" parameter.}}
#             ^^^^^^^^^^^^^


@app.get("/data/{id}", responses={404: {"description": "Not found"}})
def partially_documented_responses(id: int):
    if ...:
        raise HTTPException(status_code=400, detail="Invalid ID")  # Noncompliant {{Document this HTTPException with status code 400 in the "responses" parameter.}}
#             ^^^^^^^^^^^^^
    if ...:
        raise HTTPException(status_code=404, detail="Data not found")  # Compliant (404 in responses)



@app.get("/risky")
def exception_in_try_except_block():
    try:
        ...
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid value")  # Noncompliant

    try:
        # FP, HTTPException is caught
        raise HTTPException(status_code=422, detail="Invalid value")  # Noncompliant 
    except HTTPException:
        raise HTTPException(status_code=422, detail="Invalid value")  # Noncompliant


    try:
        # FP, HTTPException is caught
        raise HTTPException(status_code=422, detail="Invalid value")  # Noncompliant 
    except:
        raise HTTPException(status_code=422, detail="Invalid value")  # Noncompliant


@app.get("/dynamic")
def status_code_from_variable():
    status = 500
    raise HTTPException(status_code=status, detail="Error")  # Noncompliant


@app.get("/positional")
def positional_status_code_argument():
    raise HTTPException(404, detail="Not found")  # Noncompliant {{Document this HTTPException with status code 404 in the "responses" parameter.}}
#         ^^^^^^^^^^^^^


@app.get("/overflow-status")
def positional_status_code_overflow_literal():
    raise HTTPException(9223372036854775808, detail="not a representable status code")


@app.post("/create")
def post_method_undocumented():
    raise HTTPException(status_code=409, detail="Conflict")  # Noncompliant


@app.delete("/delete/{id}")
def delete_method_undocumented(id: int):
    raise HTTPException(status_code=404, detail="Not found")  # Noncompliant


@router.get("/items")
def api_router_instead_of_app():
    raise HTTPException(status_code=503, detail="Service unavailable")  # Noncompliant

# Testing following calls
def helper_with_exception(item_id: int):
    if ...:
        raise HTTPException(status_code=400, detail="Invalid ID")  # Noncompliant


@app.get("/items/{item_id}")
def endpoint_calling_helper(item_id: int):
    helper_with_exception(item_id)


def nested_helper_level_2():
    raise HTTPException(status_code=403, detail="Forbidden")  # Noncompliant


def nested_helper_level_1(id: int):
    if ...:
        raise HTTPException(status_code=400, detail="Bad ID")  # Noncompliant
    nested_helper_level_2()


@app.post("/items/{id}")
def endpoint_calling_nested_helpers(id: int):
    nested_helper_level_1(id)


@app.get("/public-and-internal")
@app.get("/internal/public-and-internal", include_in_schema=False)
def public_and_schema_excluded_routes():
    raise HTTPException(status_code=404, detail="Not found")  # Noncompliant


@router.get("/explicitly-included", include_in_schema=True)
def route_explicitly_included():
    raise HTTPException(status_code=404, detail="Not found")  # Noncompliant


@router.get("/unknown-flag", include_in_schema=some_unresolved_flag)
def route_with_unresolved_include_flag():
    raise HTTPException(status_code=404, detail="Not found")  # Noncompliant


included_router = APIRouter(include_in_schema=True)


@included_router.get("/router-explicitly-included")
def route_on_explicitly_included_router():
    raise HTTPException(status_code=404, detail="Not found")  # Noncompliant


def build_router() -> APIRouter:
    return APIRouter()


dynamic_router = build_router()


@dynamic_router.get("/dynamic")
def route_on_unresolved_router():
    raise HTTPException(status_code=404, detail="Not found")  # Noncompliant


# The receiver is a call, not a name, so there is nothing to resolve back to a constructor
@build_router().get("/inline-factory")
def route_on_inline_factory_router():
    raise HTTPException(status_code=404, detail="Not found")  # Noncompliant


# The route method is aliased, so the decorator callee is a plain name instead of a member access
aliased_route = app.get


@aliased_route("/aliased-route")
def route_on_aliased_route_method():
    raise HTTPException(status_code=404, detail="Not found")  # Noncompliant


# The router is a parameter, so its construction arguments are not reachable
def register_scoped_routes(scoped_router: APIRouter):
    @scoped_router.get("/scoped")
    def scoped_route():
        raise HTTPException(status_code=404, detail="Not found")  # Noncompliant


publicly_included_router = APIRouter()
app.include_router(publicly_included_router, prefix="/v1")


@publicly_included_router.get("/publicly-included")
def route_on_publicly_included_router():
    raise HTTPException(status_code=404, detail="Not found")  # Noncompliant


# The flag only applies to the router actually being included, not to any router mentioned in the call
misplaced_router = APIRouter()
app.include_router(APIRouter(), dependencies=[misplaced_router], include_in_schema=False)


@misplaced_router.get("/misplaced")
def route_on_router_in_another_argument():
    raise HTTPException(status_code=404, detail="Not found")  # Noncompliant


# The router is passed to an unrelated call, which carries no schema information
handed_over_router = APIRouter()
some_other_decorator(handed_over_router)


@handed_over_router.get("/handed-over")
def route_on_router_passed_to_unrelated_call():
    raise HTTPException(status_code=404, detail="Not found")  # Noncompliant


# Compliant cases

@app.get("/internal/pool-stats", include_in_schema=False)
async def schema_excluded_app_route():
    raise HTTPException(status_code=500, detail="Internal error")  # Compliant (excluded from OpenAPI schema)


@router.get("/internal/router-stats", include_in_schema=False)
def schema_excluded_router_route():
    raise HTTPException(status_code=503, detail="Service unavailable")  # Compliant (excluded from OpenAPI schema)


@app.get("/internal/falsy-number", include_in_schema=0)
def route_excluded_with_falsy_number():
    raise HTTPException(status_code=500, detail="Internal error")  # Compliant (excluded from OpenAPI schema)


@app.get("/internal/none", include_in_schema=None)
def route_excluded_with_none():
    raise HTTPException(status_code=500, detail="Internal error")  # Compliant (excluded from OpenAPI schema)


internal_router = APIRouter(include_in_schema=False)


@internal_router.get("/internal/from-router-constructor")
def route_on_schema_excluded_router():
    raise HTTPException(status_code=503, detail="Service unavailable")  # Compliant (router excluded from OpenAPI schema)


internal_app = FastAPI(include_in_schema=False)


@internal_app.get("/internal/from-app-constructor")
def route_on_schema_excluded_app():
    raise HTTPException(status_code=500, detail="Internal error")  # Compliant (app excluded from OpenAPI schema)


internal_router_alias = internal_router


@internal_router_alias.get("/internal/via-alias")
def route_on_schema_excluded_router_alias():
    raise HTTPException(status_code=503, detail="Service unavailable")  # Compliant (router excluded from OpenAPI schema)


registered_router = APIRouter()
app.include_router(registered_router, include_in_schema=False)


@registered_router.get("/internal/via-include-router")
def route_on_router_included_out_of_schema():
    raise HTTPException(status_code=503, detail="Service unavailable")  # Compliant (excluded when included in the app)


@app.get("/users2/{user_id}", responses={404: {"description": "User not found"}})
def single_status_code_documented(user_id: int):
    if ...:
        raise HTTPException(status_code=404, detail="User not found")  # Compliant


@app.put(
    "/items2/{item_id}",
    responses={
        404: {"description": "Item not found"},
        403: {"description": "Forbidden"}
    }
)
def multiple_status_codes_documented(item_id: int, item: dict):
    if ...:
        raise HTTPException(status_code=404, detail="Item not found")  # Compliant
    if ...:
        raise HTTPException(status_code=403, detail="Forbidden")  # Compliant


@app.get("/simple")
def no_exception_raised():
    ...  # Compliant


@app.get(
    "/home",
    responses={
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "example": {"detail": "Bad Request"}
                }
            }
        }
    }
)
def comprehensive_documentation_with_examples():
    raise HTTPException(status_code=400, detail="Bad Request")  # Compliant


@app.get("/items3", responses={"404": {"description": "Not found"}})
def status_code_as_string_in_responses():
    raise HTTPException(status_code=404, detail="Not found")  # Compliant


@some_other_decorator
def non_fastapi_decorator():
    raise HTTPException(status_code=404, detail="Not found")  # Compliant (not a FastAPI endpoint)


def helper_with_exception_documented(user_id: int):
    if ...:
        raise HTTPException(status_code=400, detail="Invalid ID")


@app.get("/users3/{user_id}", responses={400: {"description": "Invalid user ID"}})
def endpoint_with_documented_helper_exception(user_id: int):
    helper_with_exception_documented(user_id)  # Compliant (400 is documented)


def standalone_helper_not_called():
    raise HTTPException(status_code=500, detail="Error")  # Compliant (not used by any endpoint)


# Edge cases

@app.get("/empty", responses={})
def empty_responses_dict():
    raise HTTPException(status_code=404, detail="Not found")  # Noncompliant


my_responses = {404: {"description": "Not found"}}
@app.get("/var", responses=my_responses)
def responses_from_variable():
    raise HTTPException(status_code=404, detail="Not found")  # Compliant (cannot analyze variable)


@app.get("/multi", responses={404: {"description": "Not found"}})
@some_other_decorator
def multiple_decorators():
    raise HTTPException(status_code=404, detail="Not found")  # Compliant


class MyEndpoints:
    @app.get("/class-method")
    def class_method_endpoint(self):
        raise HTTPException(status_code=404, detail="Not found")  # Noncompliant


@router.post("/router-post")
def router_post_method():
    raise HTTPException(status_code=400, detail="Bad request")  # Noncompliant


@router.put("/router-put", responses={404: {"description": "Not found"}})
def router_put_method():
    raise HTTPException(status_code=404, detail="Not found")  # Compliant


@router.delete("/router-delete")
def router_delete_method():
    raise HTTPException(status_code=404, detail="Not found")  # Noncompliant


@router.patch("/router-patch")
def router_patch_method():
    raise HTTPException(status_code=422, detail="Unprocessable")  # Noncompliant


@router.options("/router-options")
def router_options_method():
    raise HTTPException(status_code=405, detail="Method not allowed")  # Noncompliant


@router.head("/router-head")
def router_head_method():
    raise HTTPException(status_code=404, detail="Not found")  # Noncompliant


@router.trace("/router-trace")
def router_trace_method():
    raise HTTPException(status_code=404, detail="Not found")  # Noncompliant


@app.get("/reraise")
def bare_raise_statement():
    try:
        ...
    except Exception:
        raise  # Compliant (bare raise, no HTTPException)


@app.get("/other-exception")
def raise_non_http_exception():
    raise ValueError("Not an HTTPException")  # Compliant (not HTTPException)


exception_instance = HTTPException(status_code=500, detail="Error")
@app.get("/raise-instance")
def raise_exception_instance():
    raise exception_instance  # Compliant (cannot analyze instance)


@app.get("/no-status-code")
def exception_without_status_code():
    raise HTTPException(detail="Missing status code")  # Compliant (no status_code to analyze)

@app.get("/no-status-code")
def exception_with_inner_function():
    def inner_function():
        raise HTTPException(detail="Missing status code")  # Compliant

    def inner_function2():
        raise HTTPException(status_code=500, detail="Missing status code")  # Noncompliant
    
    inner_function2()


@app.get("/no-status-code")
def call_same_function_twice_first_call():
    called_function()

@app.get("/no-status-code")
def call_same_function_twice_second_call():
    called_function()


def called_function():
    raise HTTPException(status_code=500, detail="Missing status code")  # Noncompliant


base_responses = {500: {"description": "Server error"}}
@app.get("/dict-unpacking", responses={404: {"description": "Not found"}, **base_responses})
def responses_with_dict_unpacking():
    raise HTTPException(status_code=404, detail="Not found")  # Compliant

# Dummy decorator for test
def some_other_decorator(func):
    return func

