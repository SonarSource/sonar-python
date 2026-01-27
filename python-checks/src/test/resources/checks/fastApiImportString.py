def noncompliant_examples():
    def app_with_conditional_reload():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()

        if True:
            uvicorn.run(app, reload=True)  # Noncompliant {{Pass the application as an import string when using 'reload'.}}

    def reload_true_noncompliant():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        uvicorn.run(app, debug=True, reload=False)  # Noncompliant {{Pass the application as an import string when using 'debug'.}}

    def workers_greater_than_one_noncompliant():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        uvicorn.run(app, workers=2)  # Noncompliant {{Pass the application as an import string when using 'workers'.}}

    def workers_and_debug_noncompliant():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        uvicorn.run(app, workers=3, debug=True)  # Noncompliant {{Pass the application as an import string when using 'workers' and 'debug'.}}

    # App object with reload in function context
    def reload_true_noncompliant():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        def start_server():
            uvicorn.run(app, reload=True, host="localhost")  # Noncompliant {{Pass the application as an import string when using 'reload'.}}

    def app_instance_as_variable():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        app_variable = app
        uvicorn.run(app_variable, host="0.0.0.0", port=8000, debug=True, reload=True) # Noncompliant

    def all_three_parameters_noncompliant():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        uvicorn.run(app, reload=True, workers=2, debug=True)  # Noncompliant {{Pass the application as an import string when using 'reload', 'workers', and 'debug'.}}

    def reload_via_variable():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        enable_reload = True
        uvicorn.run(app, reload=enable_reload)  # Noncompliant {{Pass the application as an import string when using 'reload'.}}

    def workers_via_variable_noncompliant():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        num_workers = 4
        uvicorn.run(app, workers=num_workers)  # Noncompliant {{Pass the application as an import string when using 'workers'.}}

def compliant_examples():
    def import_string_reload_true():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        uvicorn.run("main:app", reload=True)

    def import_string_debug_true():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        uvicorn.run("main:app", debug=True)

    def import_string_workers():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        uvicorn.run("main:app", workers=4)

    def app_object_no_debug_reload_workers():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        uvicorn.run(app, host="0.0.0.0", port=8000)

    def workers_equals_one_compliant():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        uvicorn.run(app, workers=1)

    def workers_one_via_variable_compliant():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        num_workers = 1
        uvicorn.run(app, workers=num_workers)

    def unknown_workers_compliant():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        uvicorn.run(app, workers=unknown_workers())

    def unpacked_workers_compliant():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        config = {"workers": 2}
        uvicorn.run(app, **config)

    def import_string_stored_in_variable():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()

        app_import_string = "main:app"
        uvicorn.run(app_import_string, host="0.0.0.0", port=8000, reload=True)

    def app_instance_unknown_fp():
        import uvicorn

        def unknown():
            pass

        def main():
            app_instance = unknown()
            uvicorn.run(app_instance, host="0.0.0.0", port=8000, reload=True)

    def reload_false_via_variable():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        enable_reload = False
        uvicorn.run(app, reload=enable_reload)

    def unknown_workers_value_compliant():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        def get_workers():
            return 4

        uvicorn.run(app, workers=get_workers())


def edge_cases():
    def no_arguments():
        import uvicorn
        uvicorn.run()

    def unpacking_as_first_arg():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        args = (app,)
        uvicorn.run(*args, reload=True)

    def debug_explicitly_false():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        uvicorn.run(app, debug=False, host="0.0.0.0")

    def unknown_call_debug():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        uvicorn.run(app, debug=unknown(), host="0.0.0.0")

    def unknown_variable_debug():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        uvicorn.run(app, debug=unknown_var, host="0.0.0.0")

    def reload_explicitly_false_with_workers():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        uvicorn.run(app, reload=False, workers=2)  # Noncompliant {{Pass the application as an import string when using 'workers'.}}

    def workers_equals_one_edge_case():
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        uvicorn.run(app, reload=False, workers=1)

