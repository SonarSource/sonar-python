from fastapi import FastAPI, APIRouter

app = FastAPI()
router = APIRouter()

some_other_framework = None  
custom_app = None 

def noncompliant_app_route_with_get():
    @app.route("/users", methods=["GET"])  # Noncompliant {{Replace this generic "route()" decorator with a specific HTTP method decorator.}}
#    ^^^^^^^^^
    def get_users():
        return {"users": []}

def noncompliant_app_route_with_post():
    @app.route("/users", methods=["POST"])  # Noncompliant
    def create_user(user):
        return {"user": user}

def noncompliant_app_route_with_put():
    @app.route("/items/{item_id}", methods=["PUT"])  # Noncompliant
    def update_item(item_id: int):
        return {"updated": item_id}

def noncompliant_app_route_with_delete():
    @app.route("/items/{item_id}", methods=["DELETE"])  # Noncompliant
    def delete_item(item_id: int):
        return {"deleted": item_id}

def noncompliant_app_route_with_patch():
    @app.route("/items/{item_id}", methods=["PATCH"])  # Noncompliant
    def patch_item(item_id: int):
        return {"patched": item_id}

def noncompliant_router_route_with_get():
    @router.route("/items", methods=["GET"])  # Noncompliant {{Replace this generic "route()" decorator with a specific HTTP method decorator.}}
#    ^^^^^^^^^^^^
    def list_items():
        return {"items": []}

def noncompliant_router_route_with_post():
    @router.route("/items", methods=["POST"])  # Noncompliant
    def create_item():
        return {"item": {}}

def noncompliant_route_with_options():
    @app.route("/health", methods=["OPTIONS"])  # Noncompliant
    def options_health():
        return {}

def noncompliant_route_with_head():
    @app.route("/health", methods=["HEAD"])  # Noncompliant
    def head_health():
        return {}

def noncompliant_route_with_trace():
    @app.route("/trace", methods=["TRACE"])  # Noncompliant
    def trace_endpoint():
        return {}

def noncompliant_with_lowercase_method():
    @app.route("/lower", methods=["get"])  # Noncompliant {{Replace this generic "route()" decorator with a specific HTTP method decorator.}}
#    ^^^^^^^^^
    def lower_method():
        return {}

# Edge case: path parameters in various positions
def noncompliant_with_additional_params():
    @app.route("/users/{user_id}", methods=["GET"], response_model=None)  # Noncompliant
    def get_user(user_id: int):
        return {}

def noncompliant_with_status_code():
    @app.route("/created", methods=["POST"], status_code=201)  # Noncompliant
    def create_resource():
        return {}

def compliant_specific_get_decorator():
    @app.get("/users")
    def get_users():
        return {"users": []}

def compliant_specific_post_decorator():
    @app.post("/users")
    def create_user(user):
        return {"user": user}

def compliant_specific_put_decorator():
    @app.put("/items/{item_id}")
    def update_item(item_id: int):
        return {"updated": item_id}

def compliant_specific_delete_decorator():
    @app.delete("/items/{item_id}")
    def delete_item(item_id: int):
        return {"deleted": item_id}

def compliant_specific_patch_decorator():
    @app.patch("/items/{item_id}")
    def patch_item(item_id: int):
        return {"patched": item_id}

def compliant_router_specific_decorator():
    @router.get("/items")
    def list_items():
        return {"items": []}

def compliant_options_decorator():
    @app.options("/health")
    def options_health():
        return {}

def compliant_head_decorator():
    @app.head("/health")
    def head_health():
        return {}

def compliant_trace_decorator():
    @app.trace("/trace")
    def trace_endpoint():
        return {}

def compliant_route_without_methods():
    @app.route("/catch-all")
    def catch_all():
        return {}

def compliant_route_with_multiple_methods():
    @app.route("/multi", methods=["GET", "POST"])
    def multi_method():
        return {}

def compliant_route_with_custom_http_code():
    @app.route("/multi", methods=["CUSTOM"])
    def multi_method():
        return {}

def compliant_other_framework():
    @some_other_framework.route("/other", methods=["GET"])
    def other_framework():
        return {}

def compliant_custom_route():
    @custom_app.route("/custom", methods=["GET"])
    def custom_route():
        return {}

def compliant_methods_not_list():
    @app.route("/weird", methods="GET")  # This is actually invalid FastAPI, but not our concern
    def weird_methods():
        return {}

def compliant_decorator_without_call():
    decorator_ref = app.route
    @decorator_ref
    def no_call_expression():
        return {}

def compliant_methods_list_with_number():
    @app.route("/number-method", methods=[1])
    def number_method():
        return {}

def compliant_methods_from_variable():
    methods_var = ["GET"]
    @app.route("/dynamic", methods=methods_var)  # Should be compliant - can't statically analyze
    def dynamic_method():
        return {}
