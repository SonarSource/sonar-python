from contextlib import contextmanager
import contextlib


def noncompliant_examples():
    @contextmanager
    def temporary_file(path):
        handle = open(path, "w")
        yield handle  # Noncompliant {{Cleanup after this "yield" may be skipped on early exit.}}
#       ^^^^^^^^^^^^
        handle.close()
#       ^^^^^^^^^^^^^^< {{Move this into "try"/"finally".}}

    @contextmanager
    def locked_section(lock):
        lock.acquire()
        yield  # Noncompliant {{Cleanup after this "yield" may be skipped on early exit.}}
#       ^^^^^
        lock.release()
#       ^^^^^^^^^^^^^^< {{Move this into "try"/"finally".}}

    @contextmanager
    def multiple_cleanup_statements(resource):
        resource.open()
        yield resource  # Noncompliant
#       ^^^^^^^^^^^^^^
        resource.flush()
#       ^^^^^^^^^^^^^^^^< {{Move this into "try"/"finally".}}
        resource.close()
#       ^^^^^^^^^^^^^^^^< {{Move this into "try"/"finally".}}

    @contextlib.contextmanager
    def qualified_decorator(conn):
        conn.connect()
        yield conn  # Noncompliant
        conn.close()

    @contextmanager
    def cleanup_after_nested_with(lock, handle):
        with lock:
            yield handle  # Noncompliant
#           ^^^^^^^^^^^^
        handle.close()
#       ^^^^^^^^^^^^^^< {{Move this into "try"/"finally".}}

    @contextmanager
    def cleanup_outside_try(resource):
        try:
            yield resource  # Noncompliant
#           ^^^^^^^^^^^^^^
        except Exception:
            pass
        resource.close()
#       ^^^^^^^^^^^^^^^^< {{Move this into "try"/"finally".}}

    @contextmanager
    def cleanup_after_try_with_logging_finally(resource):
        try:
            yield resource  # Noncompliant
#           ^^^^^^^^^^^^^^
        finally:
            log("exited")
        resource.close()
#       ^^^^^^^^^^^^^^^^< {{Move this into "try"/"finally".}}

    @contextmanager
    def bare_yield_then_cleanup():
        setup()
        yield  # Noncompliant
        teardown()

    class ResourceManager:
        @contextmanager
        def managed(self):
            self.acquire()
            yield self  # Noncompliant
            self.release()


from contextlib import contextmanager as cm


@cm
def aliased_decorator(path):
    f = open(path)
    yield f  # Noncompliant
    f.close()


def compliant_examples():
    @contextmanager
    def temporary_file(path):
        handle = open(path, "w")
        try:
            yield handle
        finally:
            handle.close()

    @contextmanager
    def locked_section(lock):
        with lock:
            yield

    @contextlib.contextmanager
    def only_yield():
        yield

    @contextmanager
    def yield_value_only():
        yield 42

    @contextmanager
    def nested_with_resource(path):
        with open(path, "w") as handle:
            yield handle

    @contextmanager
    def try_finally_with_bare_yield(lock):
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    @contextmanager
    def cleanup_only_in_finally(resource):
        resource.open()
        try:
            yield resource
        finally:
            resource.flush()
            resource.close()

    def not_a_context_manager(path):
        handle = open(path, "w")
        yield handle
        handle.close()

    @contextmanager
    def pass_after_yield():
        yield
        pass

    class GoodManager:
        @contextmanager
        def managed(self):
            with self.lock:
                yield self

    from contextlib import asynccontextmanager

    @asynccontextmanager
    def async_manager(resource):
        # asynccontextmanager is out of scope for this rule
        yield resource
        resource.close()


def edge_cases():
    @contextmanager
    def statement_after_try_finally(resource):
        try:
            yield resource  # Noncompliant
        finally:
            resource.release()
        log("done")

    @contextmanager
    def yield_in_if(flag, resource):
        if flag:
            yield resource  # Noncompliant
            resource.close()
        else:
            yield None

    @contextmanager
    def nested_function_ignored(resource):
        def helper():
            yield 1
            cleanup()

        try:
            yield resource
        finally:
            resource.close()

    contextmanager_alias = contextmanager

    @contextmanager_alias
    def via_variable(resource):
        yield resource  # Noncompliant
        resource.close()

    @contextmanager()
    def call_form_decorator(resource):
        received = yield resource  # Noncompliant
#                  ^^^^^^^^^^^^^^
        resource.close()
#       ^^^^^^^^^^^^^^^^< {{Move this into "try"/"finally".}}

    @contextmanager
    def assignment_yield_with_finally(resource):
        try:
            received = yield resource
        finally:
            resource.close()

    @contextmanager
    def cleanup_in_try_else(resource):
        # Success-path work in else (e.g. commit) is intentionally skipped on exceptions;
        # resource release belongs in finally. Not flagged.
        try:
            yield resource
        except Exception:
            handle_error()
        else:
            notify_success()
        finally:
            resource.close()

    @contextmanager
    def break_after_yield_with_finally(lock_fn):
        obtained = False
        try:
            while True:
                obtained = True
                yield
                break
        finally:
            if obtained:
                remove(lock_fn)

    @contextmanager
    def return_only_after_yield():
        yield
        return

    @contextmanager
    def early_return_branch_not_sibling_cleanup(detached, resource):
        # Path-insensitive sibling of the if is not cleanup for the yield+return branch
        if detached:
            yield
            return
        resource.open()
        try:
            yield resource
        finally:
            resource.close()

    @contextmanager
    def assert_statement_only_after_yield(value):
        try:
            yield value
            assert value is not None
        finally:
            cleanup(value)


import logging

log = logging.getLogger(__name__)


def logging_examples():
    @contextmanager
    def logging_only_after_yield(cursor, conn):
        # Observational logging is not cleanup; real release is in finally
        try:
            yield cursor
            log.debug("Connected")
            print("done")
        finally:
            conn.close()

    @contextmanager
    def logging_mixed_with_real_cleanup(cursor):
        yield cursor  # Noncompliant
#       ^^^^^^^^^^^^
        log.info("ok")
        cursor.close()
#       ^^^^^^^^^^^^^^< {{Move this into "try"/"finally".}}


import unittest


class SubclassAssertions(unittest.TestCase):
    """Subclass methods must be ignored like unittest.TestCase assertions."""

    @contextmanager
    def subclass_assertions_only_after_yield(self, expected):
        try:
            out = []
            yield out
            self.assertEqual(out, expected)
            self.assertIn("ok", out)
        finally:
            out.clear()


class AssertionContextManagers(unittest.TestCase):
    @contextmanager
    def assertions_only_after_yield_in_try(self, expected):
        # Real cleanup is in finally; assertions after yield are not cleanup
        try:
            out = []
            yield out
            self.assertEqual(out, expected)
            self.assertIn("ok", out)
        finally:
            out.clear()

    @contextmanager
    def assertion_mixed_with_real_cleanup(self, resource):
        yield resource  # Noncompliant
#       ^^^^^^^^^^^^^^
        self.assertIsNotNone(resource)
        resource.close()
#       ^^^^^^^^^^^^^^^^< {{Move this into "try"/"finally".}}

    @contextmanager
    def module_logging_only_after_yield(self, cursor, conn):
        try:
            yield cursor
            logging.info("connected")
        finally:
            conn.close()

    @contextmanager
    def fail_only_after_yield(self):
        try:
            yield
            self.fail("expected error")
        except AssertionError:
            raise


def success_path_and_nested_with_examples():
    @contextmanager
    def commit_on_success_with_finally(session):
        # commit is success-path; real cleanup is in finally
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def success_work_inside_nested_with(tmp_factory, upload):
        # Nested with owns tempfile cleanup; flush/upload are success-path only
        with tmp_factory() as tmp_file:
            yield tmp_file
            tmp_file.flush()
            upload(tmp_file.name)

    @contextmanager
    def cleanup_after_nested_with(worker_cm, notify):
        with worker_cm() as worker:
            yield worker  # Noncompliant
#           ^^^^^^^^^^^^
        notify(worker)
#       ^^^^^^^^^^^^^^< {{Move this into "try"/"finally".}}

    @contextmanager
    def assertion_error_inside_nested_with(sup):
        with sup:
            yield
            raise AssertionError("missing warning")

    @contextmanager
    def yield_break_then_check_in_try_finally(alloc, free):
        # Post-loop check in try body is not unprotected cleanup when finally frees
        handle = None
        try:
            while True:
                handle = alloc()
                yield handle
                break
            if handle is None:
                raise RuntimeError("missing")
        finally:
            if handle is not None:
                free(handle)

    @contextmanager
    def assertion_if_with_prelude_assignment_after_try(name):
        # Local assignments preparing an AssertionError message are not cleanup
        try:
            n_bad = 0
            yield
            n_bad = count_problems()
        finally:
            restore()
        if n_bad:
            label = " when calling %s" % name if name else ""
            raise AssertionError("problems found%s" % label)

    @contextmanager
    def deferred_reraise_after_nested_with(tx):
        # Capturing then re-raising after the nested with is not resource cleanup
        error = None
        with tx():
            try:
                yield
            except DatabaseError:
                raise
            except Exception as e:
                error = e
        if error:
            raise error

    @contextmanager
    def result_assembly_for_after_try_finally(ctx):
        # Filling the yielded container after finally is success-path result work
        ctx.enable()
        try:
            graphs = []
            yield graphs
            metadata = ctx.export()
        finally:
            ctx.disable()
        for graph in metadata.graphs:
            graphs.append(graph)

    @contextmanager
    def warning_rewrite_after_nested_with(record_warnings):
        # Re-emitting/rewriting warnings after the nested with is observational
        import warnings
        with record_warnings() as record:
            yield
        if len(record) > 0:
            match = None
            for warning in record:
                if warning.category is UserWarning:
                    category = DeprecationWarning
                    message = "rewritten"
                else:
                    category, message = warning.category, warning.message
                warnings.warn_explicit(
                    message=message,
                    category=category,
                    filename=warning.filename,
                    lineno=warning.lineno,
                )

    @contextmanager
    def pytest_fail_only_after_yield():
        import pytest
        try:
            yield
            pytest.fail("expected error")
        except AssertionError:
            raise

    @contextmanager
    def bare_reraise_after_nested_with(cm):
        with cm():
            try:
                yield
            except Exception:
                raise

    @contextmanager
    def assertion_if_with_elif_else(flag, value):
        try:
            yield value
            if flag == 1:
                assert value
            elif flag == 2:
                raise AssertionError("bad")
            else:
                x = value
                raise AssertionError(x)
        finally:
            cleanup(value)

    @contextmanager
    def state_restoring_for_after_yield(names):
        saved = dict(names)
        names.clear()
        yield  # Noncompliant
#       ^^^^^
        for key, value in saved.items():
#       ^[el=+4;ec=4]< {{Move this into "try"/"finally".}}
            names[key] = value

    @contextmanager
    def state_restoring_for_inside_if_with_logging(saved, names):
        yield  # Noncompliant
#       ^^^^^
        if saved:
#       ^[el=+6;ec=4]< {{Move this into "try"/"finally".}}
            for key, value in saved.items():
                names[key] = value
            log.info("restored")

    @contextmanager
    def guarded_jump_after_yield(items, stop):
        # A guarded break releases nothing, just like a bare one
        for item in items:
            yield item
            if stop:
                break

    @contextmanager
    def observational_while_after_yield(monitor):
        # A while is classified from its body, just like a for
        yield
        while monitor.is_active():
            log.info(monitor.status)

    @contextmanager
    def while_with_real_cleanup_after_yield(handles):
        yield  # Noncompliant
#       ^^^^^
        while handles:
#       ^[el=+4;ec=4]< {{Move this into "try"/"finally".}}
            handles.pop().close()

    @contextmanager
    def state_restoring_while_after_yield(saved, names):
        yield  # Noncompliant
#       ^^^^^
        while saved:
#       ^[el=+4;ec=4]< {{Move this into "try"/"finally".}}
            names[saved.pop()] = None

    @contextmanager
    def result_assembly_while_after_try_finally(ctx):
        # A loop right after a protected try consumes what that try produced
        ctx.enable()
        try:
            graphs = []
            yield graphs
            pending = ctx.export()
        finally:
            ctx.disable()
        while pending:
            graphs.append(pending.pop())

    @contextmanager
    def observational_for_else_after_yield(items):
        yield
        for item in items:
            pass
        else:
            raise AssertionError("empty")

    @contextmanager
    def observational_assert_inside_for_if(items):
        yield
        for item in items:
            if False:
                pass
            elif item:
                assert item
            else:
                raise AssertionError("missing")

    @contextmanager
    def for_with_only_pass_after_yield(xs):
        # pass-only for is not observational cleanup-skip; still reported
        yield  # Noncompliant
#       ^^^^^
        for x in xs:
#       ^[el=+4;ec=4]< {{Move this into "try"/"finally".}}
            pass

    @contextmanager
    def for_else_with_real_cleanup(items, resource):
        yield  # Noncompliant
#       ^^^^^
        for item in items:
#       ^[el=+6;ec=4]< {{Move this into "try"/"finally".}}
            assert item
        else:
            resource.close()

    @contextmanager
    def bare_raise_after_yield_in_try():
        try:
            yield
            raise
        except Exception:
            pass

    @contextmanager
    def named_reraise_sibling(error):
        yield
        raise error

    @contextmanager
    def raise_of_any_type_after_yield(r):
        # Raising signals a problem, it never releases a resource, whatever the exception type
        yield r
        raise RuntimeError("boom")

    @contextmanager
    def guarded_raise_of_any_type_after_yield(r, bad):
        yield r
        if bad:
            raise RuntimeError("boom")

    @contextmanager
    def assertion_prep_split_across_branches(cond, value, label):
        # A branch writing only a local prepares a value, so it does not make the selection state restoration
        yield
        if cond:
            label = value
        else:
            raise AssertionError(label)

    @contextmanager
    def attribute_restore_split_across_branches(self, resource):
        yield  # Noncompliant
#       ^^^^^
        if self.nested:
#       ^[el=+7;ec=4]< {{Move this into "try"/"finally".}}
            self.nested -= 1
        else:
            assert self.current
            self.current = None

    @contextmanager
    def subscript_restore_split_across_branches(cond, names, key, saved):
        yield  # Noncompliant
#       ^^^^^
        if cond:
#       ^[el=+6;ec=4]< {{Move this into "try"/"finally".}}
            names[key] = saved
        else:
            log.info("kept")

    @contextmanager
    def raise_alongside_real_cleanup(r):
        yield r  # Noncompliant
#       ^^^^^^^
        r.close()
#       ^^^^^^^^^< {{Move this into "try"/"finally".}}
        raise RuntimeError("boom")

    @contextmanager
    def assignment_only_if_with_nonassignment_elif(flag, resource):
        yield  # Noncompliant
#       ^^^^^
        if flag:
#       ^[el=+6;ec=4]< {{Move this into "try"/"finally".}}
            x = 1
        elif flag == 2:
            resource.close()

    @contextmanager
    def assignment_only_if_with_nonassignment_else(flag, resource):
        yield  # Noncompliant
#       ^^^^^
        if flag:
#       ^[el=+6;ec=4]< {{Move this into "try"/"finally".}}
            x = 1
        else:
            resource.close()

    @contextmanager
    def nested_assignment_if_else_not_scaffolding(flag, resource):
        # Outer if is observational; nested if/else is not assignment-only scaffolding
        yield  # Noncompliant
#       ^^^^^
        if True:
#       ^[el=+8;ec=4]< {{Move this into "try"/"finally".}}
            assert True
            if flag:
                x = 1
            else:
                resource.close()

    @contextmanager
    def nested_function_and_lambda_ignored(resource):
        def helper():
            yield 1
            cleanup()

        unused = lambda: cleanup()
        try:
            yield resource
        finally:
            resource.close()

    @contextmanager
    def multi_expression_statement_not_ignorable(resource):
        yield resource  # Noncompliant
#       ^^^^^^^^^^^^^^
        a, b = 1, 2
#       ^^^^^^^^^^^< {{Move this into "try"/"finally".}}
        resource.close()
#       ^^^^^^^^^^^^^^^^< {{Move this into "try"/"finally".}}

    @contextmanager
    def annotated_and_compound_assignment_in_assertion_if(n):
        try:
            yield
        finally:
            restore()
        if n:
            label: str = "x"
            label += "y"
            raise AssertionError(label)

    @contextmanager
    def non_call_expression_statement(resource):
        yield resource  # Noncompliant
#       ^^^^^^^^^^^^^^
        resource
#       ^^^^^^^^< {{Move this into "try"/"finally".}}
        resource.close()
#       ^^^^^^^^^^^^^^^^< {{Move this into "try"/"finally".}}


def match_statement_examples():
    @contextmanager
    def guarded_jump_match_after_yield(items, verdict):
        # A match that only jumps releases nothing, like an if that only jumps
        for item in items:
            yield item
            match verdict:
                case "stop":
                    break
                case _:
                    continue

    @contextmanager
    def observational_match_after_yield(state):
        yield
        match state:
            case "quiet":
                pass
            case _:
                log.info("done")

    @contextmanager
    def match_with_real_cleanup_after_yield(state, resource):
        yield  # Noncompliant
#       ^^^^^
        match state:
#       ^[el=+7;ec=4]< {{Move this into "try"/"finally".}}
            case "keep":
                log.info("kept")
            case _:
                resource.close()

    @contextmanager
    def value_preparing_match_inside_assertion_if(kind, n):
        # A match assigning on every case prepares a local value, so the if stays observational
        yield
        if n:
            match kind:
                case "a":
                    label = "first"
                case _:
                    label = "other"
            raise AssertionError(label)
