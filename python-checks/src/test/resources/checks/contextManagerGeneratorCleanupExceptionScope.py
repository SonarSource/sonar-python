from contextlib import asynccontextmanager, contextmanager


@contextmanager
def assert_raises(expected_type, expected_message):
    try:
        yield
    except BaseException as exc:
        assert type(exc) is expected_type
        assert expected_message in str(exc)
    else:
        assert False, "Expected an exception"


@contextmanager
def defer_to_coordinator(coordinator):
    try:
        yield
    except:
        coordinator.request_stop()


@asynccontextmanager
async def async_suppression():
    try:
        yield
    except BaseException:
        pass


@contextmanager
def manager_already_received_exception(manager):
    # The manager's __exit__ receives the exception before this handler runs.
    try:
        with manager:
            yield
    except BaseException:
        pass


@contextmanager
def hides_exception_from_active_manager(manager):
    with manager:
        try:
            yield
        except BaseException:  # Noncompliant {{Swallowing this exception may skip cleanup of other context managers.}}
            pass


@asynccontextmanager
async def hides_exception_from_active_async_manager(manager):
    async with manager:
        try:
            yield
        except:  # Noncompliant {{Swallowing this exception may skip cleanup of other context managers.}}
            pass


@contextmanager
def own_cleanup_still_needs_protection(resource):
    try:
        yield resource  # Noncompliant {{Cleanup after this "yield" may be skipped on early exit.}}
        resource.close()
    except BaseException:
        pass


def outer_manager_is_not_active_for_nested_generator(manager):
    with manager:
        @contextmanager
        def deferred_generator():
            try:
                yield
            except BaseException:
                pass
        return deferred_generator


@contextmanager
def bound_resource_used_on_success(manager):
    with manager as resource:
        try:
            yield resource
            resource.record_success()
        except Exception:
            raise


@contextmanager
def same_spelling_in_another_scope_is_not_the_bound_resource(manager, cleanup):
    with manager as resource:
        try:
            yield  # Noncompliant {{Cleanup after this "yield" may be skipped on early exit.}}
            cleanup(lambda resource: resource)
        except Exception:
            raise
