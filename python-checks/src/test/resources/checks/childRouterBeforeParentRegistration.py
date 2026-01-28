from fastapi import FastAPI, APIRouter


app = FastAPI()
parent_router = APIRouter()
child_router = APIRouter()

app.include_router(parent_router, prefix="/api")
parent_router.include_router(child_router)  # Noncompliant {{Include child routers before registering the parent router.}}

def noncompliant_basic_parent_included_before_child():
    app = FastAPI()
    parent_router = APIRouter()
    child_router = APIRouter()

    app.include_router(parent_router, prefix="/api")
    parent_router.include_router(child_router)  # Noncompliant {{Include child routers before registering the parent router.}}


def noncompliant_nested_routers_multiple_levels():
    app = FastAPI()
    api_router = APIRouter()
    v1_router = APIRouter()
    users_router = APIRouter()

    app.include_router(api_router)
    api_router.include_router(v1_router, prefix="/v1")  # Noncompliant
    v1_router.include_router(users_router, prefix="/users")  # Noncompliant


def noncompliant_keyword_argument():
    app = FastAPI()
    router_a = APIRouter()
    router_b = APIRouter()
    app.include_router(router=router_a)
    router_a.include_router(router=router_b)  # Noncompliant


def noncompliant_in_conditional():
    app = FastAPI()
    conditional_router = APIRouter()
    conditional_child = APIRouter()
    if True:
        app.include_router(conditional_router)
    conditional_router.include_router(conditional_child)  # Noncompliant


def compliant_correct_order_bottom_up():
    app = FastAPI()
    child = APIRouter()
    parent = APIRouter()
    parent.include_router(child)
    app.include_router(parent, prefix="/ok")


def compliant_nested_correct_order():
    users = APIRouter()
    v1 = APIRouter()
    api = APIRouter()
    app = FastAPI()

    v1.include_router(users, prefix="/users")
    api.include_router(v1, prefix="/v1")
    app.include_router(api)


def compliant_independent_routers():
    app = FastAPI()
    independent1 = APIRouter()
    independent2 = APIRouter()
    app.include_router(independent1)
    app.include_router(independent2)


def compliant_single_router():
    app = FastAPI()
    single_router = APIRouter()
    app.include_router(single_router)


def compliant_router_with_routes_only():
    router = APIRouter()
    @router.get("/items")
    def get_items():
        return {"items": []}


def noncompliant_multiple_apps_shared_router():
    app_a = FastAPI()
    app_b = FastAPI()
    shared_router = APIRouter()
    child = APIRouter()

    app_a.include_router(shared_router)
    shared_router.include_router(child)  # Noncompliant


def compliant_registration_in_nested_function():
    app = FastAPI()
    router = APIRouter()
    child = APIRouter()

    def register_router():
        app.include_router(router)

    router.include_router(child)


def compliant_router_used_in_other_context():
    app = FastAPI()
    router = APIRouter()
    child = APIRouter()

    # Router is referenced but not registered yet
    print(router)
    x = router
    router.include_router(child)
    app.include_router(router)


def compliant_router_registered_after_on_same_line():
    app = FastAPI()
    router = APIRouter()
    child = APIRouter()

    router.include_router(child)
    app.include_router(router)


def noncompliant_with_keyword_arg_router():
    app = FastAPI()
    parent_router = APIRouter()
    child_router = APIRouter()

    app.include_router(router=parent_router)
    parent_router.include_router(router=child_router)  # Noncompliant


def compliant_router_as_second_positional_arg():
    # Edge case: router passed in wrong position (not first arg, not "router" keyword)
    # Should not be detected as registration
    app = FastAPI()
    router = APIRouter()
    child = APIRouter()

    router.include_router(child)
    # some_other_function(prefix="/api", router)  # Not include_router
    app.include_router(router)


def compliant_no_receiver_symbol():
    # Edge case: include_router called without qualified receiver
    app = FastAPI()
    router = APIRouter()

    # Call where we can't extract receiver symbol
    # (This is theoretical - in practice FastAPI requires qualified calls)
    router.include_router(APIRouter())


def compliant_router_in_different_scopes():
    app = FastAPI()
    router = APIRouter()
    child = APIRouter()

    def scope1():
        app.include_router(router)

    def scope2():
        # Different function scope - should not trigger
        router.include_router(child)


def compliant_router_not_as_argument():
    app = FastAPI()
    router = APIRouter()
    child = APIRouter()

    # Router used but not as an argument to include_router
    if router:
        print("router exists")

    router.include_router(child)
    app.include_router(router)


def compliant_router_in_non_include_router_call():
    app = FastAPI()
    router = APIRouter()
    child = APIRouter()

    # Router used as argument to a different function (not include_router)
    some_other_function(router)

    router.include_router(child)
    app.include_router(router)


def some_other_function(r):
    pass

def fp_with_control_flow():
    app = FastAPI()
    parent_router = APIRouter()
    child_router = APIRouter()
    if True:
        app.include_router(parent_router, prefix="/api")
    else:
        parent_router.include_router(child_router) # Noncompliant

def coverage():
    app = FastAPI()
    router = APIRouter()
    child = APIRouter()

    stored_include_router = app.include_router

    stored_include_router(router)
    
    unrelated_obj = ...
    child.include_router(unrelated_obj)
