def compliant_examples():
    # Explicit return None is compliant
    def basic_return_non():
        from fastapi import FastAPI

        app = FastAPI()

        @app.delete("/item/{id}", status_code=204)
        def delete_item(id: int):
            return None

    # Explicit Response object with 204 status is compliant
    def return_empty_response():
        from fastapi import FastAPI, Response

        app = FastAPI()

        @app.put("/item/{id}/disable", status_code=204)
        def disable_item(id: int):
            return Response(status_code=204)

    # Response object with explicit empty content is compliant
    def return_response_content_empty_string():
        from fastapi import FastAPI, Response

        app = FastAPI()

        @app.patch("/item/{id}/archive", status_code=204)
        def archive_item(id: int):
            # Archive logic here
            return Response(status_code=204, content="")

    # Explicit return None in all code paths is compliant
    def return_non_multiple_paths():
        from fastapi import FastAPI

        app = FastAPI()

        @app.delete("/item/{id}", status_code=204)
        def delete_item(id: int):
            if id > 0:
                return None
            else:
                return None

    def return_response_variable():
        from fastapi import FastAPI, Response

        app = FastAPI()

        @app.post("/item/{id}/reset", status_code=204)
        def reset_item(id: int):
            # Reset logic here
            response = Response(status_code=204)
            return response  # Compliant - Response instance with 204 status

    def overrides_status_code_exception():
        from fastapi import FastAPI, HTTPException

        app = FastAPI()

        @app.delete("/resource/{id}", status_code=204)
        def delete_resource(id: int):
            try:
                perform_deletion(id)
                return None
            except ResourceNotFound:
                raise HTTPException(status_code=404, detail="Resource not found")

    def overrides_status_code_exception_2():
        from fastapi import FastAPI, Response, HTTPException

        app = FastAPI()

        @app.patch("/resource/{id}/disable", status_code=204)
        def disable_resource(id: int):
            if not validate_permissions(id):
                raise HTTPException(status_code=403, detail="Forbidden")

            perform_disable(id)
            return Response(status_code=204)

    # Compliant: None stored in a variable
    def return_none_variable():
        from fastapi import FastAPI

        app = FastAPI()

        @app.delete("/item/{id}", status_code=204)
        def delete_item(id: int):
            result = None
            return result

    # Accepted FP: Returning result of a method that returns None
    def return_method_call_result_fp():
        from fastapi import FastAPI

        app = FastAPI()

        def perform_delete_returning_none(id: int):
            return None

        @app.delete("/item/{id}", status_code=204)
        def delete_item(id: int):
            return perform_delete_returning_none(id)  # Noncompliant {{Return an empty body for this endpoint returning 204 status.}}

    # Using 'pass' results in implicit None return, which is compliant
    def implicit_none_return():
        from fastapi import FastAPI

        app = FastAPI()

        @app.delete("/item/{id}", status_code=204)
        def delete_item(id: int):  # Compliant - implicit None return
            pass

    # No explicit return statement results in implicit None return, which is compliant
    def implicit_none_return_2():
        from fastapi import FastAPI

        app = FastAPI()

        @app.delete("/item/{id}", status_code=204)
        def delete_item(id: int):  # Compliant - implicit None return
            item_deleted = True

    def unknown_status_code():
        from fastapi import FastAPI

        app = FastAPI()

        @app.delete("/item/{id}", status_code=unknown())
        def delete_item(id: int):
            return get_deletion_result(id)

    def nested_functions_no_fp():
        from fastapi import FastAPI

        app = FastAPI()

        @app.delete("/item/{id}", status_code=204)
        def delete_item(id: int):
            def nested():
                return 42
            return None

    def empty_return():
        from fastapi import FastAPI

        app = FastAPI()

        @app.delete("/item/{id}", status_code=204)
        def delete_item(id: int):
            return


def noncompliant_examples():
    def return_from_method_call():
        from fastapi import FastAPI

        app = FastAPI()

        def get_deletion_result(id: int):
            # Unknown return value
            if id > 0:
                return None
            else:
                return "deleted"

        @app.delete("/item/{id}", status_code=204)
        def delete_item(id: int):
            # Method with uncertain return value
            return get_deletion_result(id)  # Noncompliant {{Return an empty body for this endpoint returning 204 status.}}

    def unknown_variable_returned():
        from fastapi import FastAPI

        app = FastAPI()

        @app.delete("/item/{id}", status_code=204)
        def delete_item(id: int):
            result = perform_delete(id)  # perform_delete might return anything
            if result:
                return result  # Noncompliant {{Return an empty body for this endpoint returning 204 status.}}
            else:
                return None  # This path is compliant

    def unknown_variable_returned_2():
        from fastapi import FastAPI

        app = FastAPI()

        @app.delete("/item/{id}", status_code=204)
        def delete_item(id: int):
            result = perform_delete(id)
            return result  # Noncompliant {{Return an empty body for this endpoint returning 204 status.}}

    def return_dictionary():
        from fastapi import FastAPI

        app = FastAPI()

        @app.put("/item/{id}/update", status_code=204)
        def update_item(id: int, data: dict):
            success = update_database(id, data)
            return {"updated": success}  # Noncompliant {{Return an empty body for this endpoint returning 204 status.}}

    def return_content_response():
        from fastapi import FastAPI, Response

        app = FastAPI()

        @app.post("/item/{id}/reset", status_code=204)
        def reset_item(id: int):
            # Reset logic here
            response = Response(content="Hello", status_code=204)
            return response  # Noncompliant

    def unknown_string_noncompliant():
        from fastapi import FastAPI, Response

        app = FastAPI()

        @app.post("/item/{id}/reset", status_code=204)
        def reset_item(id: int):
            # Reset logic here
            response = Response(content=unknown(), status_code=204)
            return response  # Noncompliant

    def return_tuple_noncompliant():
        from fastapi import FastAPI, Response

        app = FastAPI()

        @app.post("/item/{id}/reset", status_code=204)
        def reset_item(id: int):
            # Reset logic here
            response = Response(content="Hello", status_code=204)
            return 1, 2  # Noncompliant

    def conditional_assignment_noncompliant_locations():
        from fastapi import FastAPI, Response

        app = FastAPI()

        @app.delete("/item/{id}", status_code=204)
        def delete_item(id: int):
            if id > 0:
                response = Response(content="Deleted successfully", status_code=204)
            #              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^>
            else:
                response = Response(content="Invalid ID", status_code=204)
            #              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^>
            return response  # Noncompliant
    #       ^^^^^^^^^^^^^^^

    # Conditional assignment - mixed valid and invalid Response
    def conditional_assignment_mixed_locations():
        from fastapi import FastAPI, Response

        app = FastAPI()

        @app.put("/item/{id}", status_code=204)
        def update_item(id: int):
            if id > 0:
                response = Response(status_code=204)  # Valid
            else:
                response = Response(content="Error", status_code=204)  # Invalid
            #              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^> {{Response is assigned here}}
            return response  # Noncompliant {{Return an empty body for this endpoint returning 204 status.}}
        #   ^^^^^^^^^^^^^^^

    @some_decorator
    def function_with_other_decorator():
        ...

    def missing_status_code():
        from fastapi import FastAPI

        app = FastAPI()

        @app.delete("/item/{id}")
        def delete_item(id: int):
            return {"deleted": True}


def non_fastapi_apps_compliant():
    def non_fastapi_method():
        class FakeApp:
            def delete(self, path, status_code=None):
                def decorator(func):
                    return func
                return decorator

        app = FakeApp()

        @app.delete("/item/{id}", status_code=204)
        def delete_item(id: int):
            return {"deleted": True}

    def custom_http_methods():
        class MyCustomApp:
            def get(self, route, **kwargs):
                def wrapper(f):
                    return f
                return wrapper

            def post(self, route, **kwargs):
                def wrapper(f):
                    return f
                return wrapper

        app = MyCustomApp()

        @app.get("/data", status_code=204)
        def get_data():
            # Compliant - not a FastAPI app
            return {"data": "value"}

        @app.post("/item", status_code=204)
        def create_item():
            # Compliant - not a FastAPI app
            return "created"

    def flask_app_out_of_scope():
        from flask import Flask

        app = Flask(__name__)

        # Flask doesn't use decorators with status_code parameter like FastAPI
        @app.route("/item/<int:id>", methods=["DELETE"])
        def delete_item(id):
            # Compliant - Flask app, not FastAPI
            return {"deleted": id}, 204  # Flask returns status as tuple

    # Test case: Plain Python object named 'app' with methods
    def plain_python():
        app = object()
        # Can't decorate with a plain object's attributes
        # This is to show that name-based detection alone could be fooled

        # If someone creates a function called delete on a regular object
        if hasattr(app, 'delete'):
            # This would never actually work as a decorator
            pass

    # Accepted FN: Starlette app with JSONResponse
    def starlette_response_out_of_scope():
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        async def delete_item(request):
            # Compliant - Starlette app, not FastAPI
            # Even though it returns JSON, it's not a FastAPI endpoint
            return JSONResponse({"deleted": True}, status_code=204)

        app = Starlette(routes=[
            Route('/item/{id}', delete_item, methods=["DELETE"])
        ])
