from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware as StarletteCORS


def noncompliant_basic_cors_followed_by_middleware():
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"])  # Noncompliant {{Add CORSMiddleware last in the middleware chain.}}
#   ^^^^^^^^^^^^^^^^^^
    app.add_middleware(GZipMiddleware)


def noncompliant_multiple_middleware_after_cors():
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"])  # Noncompliant
    app.add_middleware(GZipMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])


def noncompliant_cors_in_middle_of_chain():
    app = FastAPI()
    app.add_middleware(GZipMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=["*"])  # Noncompliant
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])


def noncompliant_with_keyword_argument():
    app = FastAPI()
    app.add_middleware(middleware_class=CORSMiddleware, allow_origins=["*"])  # Noncompliant
    app.add_middleware(GZipMiddleware)


def noncompliant_starlette_app():
    app = Starlette()
    app.add_middleware(CORSMiddleware, allow_origins=["*"])  # Noncompliant
    app.add_middleware(GZipMiddleware)


def noncompliant_starlette_cors():
    app = FastAPI()
    app.add_middleware(StarletteCORS, allow_origins=["*"])  # Noncompliant
    app.add_middleware(GZipMiddleware)


def noncompliant_mixed_positional_and_keyword():
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_credentials=True)  # Noncompliant
    app.add_middleware(middleware_class=GZipMiddleware)


def noncompliant_in_conditional():
    app = FastAPI()
    if True:
        app.add_middleware(CORSMiddleware, allow_origins=["*"])  # Noncompliant
    app.add_middleware(GZipMiddleware)


def noncompliant_global_app():
    global_app.add_middleware(CORSMiddleware, allow_origins=["*"])  # Noncompliant
    global_app.add_middleware(GZipMiddleware)


def compliant_cors_is_last_middleware():
    app = FastAPI()
    app.add_middleware(GZipMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=["*"])


def compliant_cors_only_middleware():
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"])


def compliant_no_cors():
    app = FastAPI()
    app.add_middleware(GZipMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])


def noncompliant_multiple_apps():
    app1 = FastAPI()
    app2 = FastAPI()
    app1.add_middleware(CORSMiddleware, allow_origins=["*"])  # Noncompliant
    app1.add_middleware(GZipMiddleware)
    app2.add_middleware(GZipMiddleware)
    app2.add_middleware(CORSMiddleware, allow_origins=["*"])


def compliant_multiple_apps_correct_order():
    app1 = FastAPI()
    app2 = FastAPI()
    app1.add_middleware(GZipMiddleware)
    app1.add_middleware(CORSMiddleware, allow_origins=["*"])
    app2.add_middleware(GZipMiddleware)
    app2.add_middleware(CORSMiddleware, allow_origins=["*"])


def compliant_correct_order_multiple_middleware():
    app = FastAPI()
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(GZipMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=["*"])


def compliant_no_middleware():
    app = FastAPI()


def compliant_cors_added_on_same_line_is_last():
    app = FastAPI()
    app.add_middleware(GZipMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=["*"])


def compliant_different_scopes():
    app = FastAPI()

    def scope1():
        app.add_middleware(CORSMiddleware, allow_origins=["*"])

    def scope2():
        app.add_middleware(GZipMiddleware)


def compliant_middleware_before_cors_in_different_function():
    app = FastAPI()

    def register_middleware():
        app.add_middleware(GZipMiddleware)

    app.add_middleware(CORSMiddleware, allow_origins=["*"])


def noncompliant_fp_conditional_same_scope():
    app = FastAPI()
    if True:
        app.add_middleware(CORSMiddleware, allow_origins=["*"])  # Noncompliant
    else:
        app.add_middleware(GZipMiddleware)


def non_compliant_cors_from_different_variable():
    app = FastAPI()
    cors_class = CORSMiddleware
    app.add_middleware(cors_class, allow_origins=["*"]) # Noncompliant
    app.add_middleware(GZipMiddleware)


def compliant_unrelated_calls_to_app():
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"])
    app.router.add_route("/test", lambda: None)


def coverage_no_receiver():
    add_middleware(CORSMiddleware, allow_origins=["*"])


def coverage_non_name_receiver():
    app = FastAPI()
    get_app().add_middleware(CORSMiddleware, allow_origins=["*"])
    get_app().add_middleware(GZipMiddleware)


def coverage_no_arguments():
    app = FastAPI()
    app.add_middleware()


def coverage_not_cors_middleware():
    app = FastAPI()
    app.add_middleware(GZipMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])


def get_app():
    return FastAPI()


global_app = FastAPI()


def compliant_cors_type_matched_directly():
    app = FastAPI()
    app.add_middleware(GZipMiddleware)
    from fastapi.middleware.cors import CORSMiddleware as DirectCORS
    app.add_middleware(DirectCORS, allow_origins=["*"])


def compliant_middleware_not_on_name_receiver():
    get_app().add_middleware(CORSMiddleware, allow_origins=["*"])


def compliant_non_name_expression_as_middleware():
    app = FastAPI()
    middleware_dict = {"cors": CORSMiddleware}
    app.add_middleware(middleware_dict["cors"], allow_origins=["*"])

def fn_non_name_expression_as_middleware():
    app = FastAPI()
    middleware_dict = {"gzip": GZipMiddleware, "cors": CORSMiddleware}
    app.add_middleware(middleware_dict["gzip"], allow_origins=["*"])
    app.add_middleware(middleware_dict["cors"], allow_origins=["*"])

def compliant_lambda_scope():
    app = FastAPI()
    register = lambda: app.add_middleware(CORSMiddleware, allow_origins=["*"])
    app.add_middleware(GZipMiddleware)


def compliant_usage_not_receiver_of_qualified_expr():
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"])
    other_var = app
    result = other_var


def compliant_qualified_not_callee():
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"])
    method_ref = app.add_middleware


def noncompliant_cors_with_lowercase_name():
    app = FastAPI()
    from starlette.middleware.cors import CORSMiddleware as CorsMiddleware
    app.add_middleware(CorsMiddleware, allow_origins=["*"])  # Noncompliant
    app.add_middleware(GZipMiddleware)

file_level_app = FastAPI()
file_level_app.add_middleware(CORSMiddleware, allow_origins=["*"])  # Noncompliant
file_level_app.add_middleware(GZipMiddleware)

def non_compliant_cors_via_module_access():
    app = FastAPI()
    import fastapi.middleware.cors
    app.add_middleware(fastapi.middleware.cors.CORSMiddleware, allow_origins=["*"]) # Noncompliant
    app.add_middleware(GZipMiddleware)


def non_compliant_cors_via_attribute_not_recognized():
    app = FastAPI()
    import fastapi.middleware.cors as cors_module
    app.add_middleware(cors_module.CORSMiddleware, allow_origins=["*"]) # Noncompliant
    app.add_middleware(GZipMiddleware)


def compliant_app_usage_as_name_not_in_qualified():
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"])
    result = app


def noncompliant_starlette_imported_cors():
    from starlette.middleware.cors import CORSMiddleware as StarletteCorsClass
    app = Starlette()
    app.add_middleware(StarletteCorsClass, allow_origins=["*"])  # Noncompliant
    app.add_middleware(GZipMiddleware)


def compliant_cors_in_class_method():
    class AppConfig:
        def setup(self):
            app = FastAPI()
            app.add_middleware(GZipMiddleware)
            app.add_middleware(CORSMiddleware, allow_origins=["*"])


def compliant_cors_direct_import_same_name():
    from fastapi.middleware.cors import CORSMiddleware
    app = FastAPI()
    app.add_middleware(GZipMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=["*"])


def non_compliant_middleware_variable_no_cors_in_name():
    from fastapi.middleware.cors import CORSMiddleware as CORSType
    app: FastAPI = FastAPI()
    middleware = CORSType
    app.add_middleware(middleware, allow_origins=["*"]) # Noncompliant
    app.add_middleware(GZipMiddleware)
