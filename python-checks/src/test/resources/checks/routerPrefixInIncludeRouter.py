from fastapi import APIRouter, FastAPI

# --- Noncompliant Cases ---

def noncompliant_basic():
    """Basic case: prefix in include_router instead of APIRouter constructor"""
    router = APIRouter()
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")  # Noncompliant {{Define the prefix in the "APIRouter" constructor instead of in "include_router()".}}
#                              ^^^^^^

def noncompliant_router_to_router():
    """Prefix when adding router to another router"""
    parent = APIRouter()
    child = APIRouter()
    parent.include_router(child, prefix="/users")  # Noncompliant
#                                ^^^^^^

def noncompliant_keyword_argument():
    """Using keyword argument for router parameter"""
    router = APIRouter()
    app = FastAPI()
    app.include_router(router=router, prefix="/api")  # Noncompliant
#                                     ^^^^^^

def noncompliant_multiple_routers():
    """Multiple routers with prefix in include_router"""
    router1 = APIRouter()
    router2 = APIRouter()
    app = FastAPI()
    app.include_router(router1, prefix="/v1")  # Noncompliant
#                               ^^^^^^
    app.include_router(router2, prefix="/v2")  # Noncompliant
#                               ^^^^^^

def noncompliant_prefix_with_tags():
    """Prefix combined with other arguments"""
    router = APIRouter()
    app = FastAPI()
    app.include_router(router, prefix="/api", tags=["api"])  # Noncompliant
#                              ^^^^^^

# --- Compliant Cases ---

def compliant_prefix_in_constructor():
    router = APIRouter(prefix="/api/v1")
    app = FastAPI()
    app.include_router(router)  # Compliant

def compliant_no_prefix():
    router = APIRouter()
    app = FastAPI()
    app.include_router(router)  # Compliant

def compliant_both_prefixes():
    """Prefix in both places - still compliant (user's choice to combine)"""
    router = APIRouter(prefix="/api")
    app = FastAPI()
    app.include_router(router, prefix="/v1")  # Compliant

def compliant_prefix_in_constructor_with_tags():
    """Prefix in constructor, other args in include_router"""
    router = APIRouter(prefix="/api")
    app = FastAPI()
    app.include_router(router, tags=["api"])  # Compliant

def compliant_only_tags():
    """Only tags, no prefix"""
    router = APIRouter()
    app = FastAPI()
    app.include_router(router, tags=["api"])  # Compliant

def fn_router_without_symbol():
    """Router created inline - edge case"""
    app = FastAPI()
    app.include_router(APIRouter(), prefix="/api")  # FN 

def fn_unknown_router_origin():
    """Router comes from function call"""
    app = FastAPI()
    app.include_router(get_router(), prefix="/api")  # FN 

def fn_router_from_import():
    """Router imported from another module"""
    from other_module import router
    app = FastAPI()
    app.include_router(router, prefix="/api")  # FN 

# --- Edge Cases ---

def edge_case_reassigned_router():
    """Router reassigned to another variable"""
    router = APIRouter()
    r = router
    app = FastAPI()
    app.include_router(r, prefix="/api")  # Noncompliant

def edge_case_conditional():
    router = APIRouter()
    app = FastAPI()
    if True:
        app.include_router(router, prefix="/api")  # Noncompliant
#                                  ^^^^^^

def edge_case_annotated_assignment():
    router: APIRouter = APIRouter()
    app = FastAPI()
    app.include_router(router, prefix="/api")  # Noncompliant
#                              ^^^^^^

def edge_case_router_as_function_parameter():
    def setup_routes(router: APIRouter):
        app = FastAPI()
        app.include_router(router, prefix="/api")  # potential FN 

def edge_case_multiple_assignments():
    router = APIRouter()
    router = APIRouter(prefix="/api")  
    app = FastAPI()
    app.include_router(router, prefix="/v1")  # FN

def edge_case_no_router():
    router = APIRouter()
    app = FastAPI()
    app.include_router(prefix="/api", router) 
