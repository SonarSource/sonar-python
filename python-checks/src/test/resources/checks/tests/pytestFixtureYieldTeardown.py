import pytest


@pytest.fixture
def with_finalizer(request):
    resource = open_resource()
    request.addfinalizer(resource.close)  # Noncompliant {{Replace this "request.addfinalizer()" call with a yield-based teardown.}}
#   ^^^^^^^^^^^^^^^^^^^^
    return resource


@pytest.fixture(scope="module")
def with_finalizer_and_scope(request):
    resource = open_resource()
    request.addfinalizer(resource.close)  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^
    return resource


@pytest.fixture
def finalizer_without_return(request):
    setup()
    request.addfinalizer(teardown)  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^


@pytest.fixture
def several_finalizers(request):
    # Multi-resource staged addfinalizer is a legitimate pytest pattern for partial-failure
    # safety (cleanup already registered if a later connect raises). Still reported to stay
    # aligned with Ruff PT021; suppress intentionally when that tradeoff is deliberate.
    first = open_resource()
    second = open_resource()
    request.addfinalizer(first.close)  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^
    request.addfinalizer(second.close)  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^
    return first, second


@pytest.fixture
def finalizer_in_nested_block(request, flag):
    resource = open_resource()
    if flag:
        request.addfinalizer(resource.close)  # Noncompliant
#       ^^^^^^^^^^^^^^^^^^^^
    return resource


@pytest.fixture
def finalizer_with_lambda(request):
    resource = open_resource()
    request.addfinalizer(lambda: resource.close())  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^
    return resource


@pytest.fixture
def bare_yield_with_finalizer(request):
    resource = open_resource()
    request.addfinalizer(resource.close)  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^
    yield


@pytest.fixture
def yield_teardown():
    resource = open_resource()
    yield resource
    resource.close()


@pytest.fixture
def yield_and_finalizer(request):
    resource = open_resource()
    request.addfinalizer(resource.close)
    yield resource


@pytest.fixture
def factory_as_fixture(request):
    created = []

    def make_resource(name):
        resource = open_resource(name)
        request.addfinalizer(resource.close)
        created.append(resource)
        return resource

    return make_resource


@pytest.fixture
def factory_as_lambda(request):
    return lambda name: request.addfinalizer(open_resource(name).close)


@pytest.fixture
def no_teardown_at_all():
    return open_resource()


def test_something(request):
    resource = open_resource()
    request.addfinalizer(resource.close)
    assert resource is not None


def helper_registering_finalizer(request):
    request.addfinalizer(cleanup)


@pytest.fixture
def finalizer_on_other_object(monkeypatch):
    monkeypatch.addfinalizer(cleanup)
    return monkeypatch


@pytest.fixture
def other_request_method(request):
    return request.getfixturevalue("with_finalizer")


class TestResources:

    @pytest.fixture
    def fixture_in_class(self, request):
        resource = open_resource()
        request.addfinalizer(resource.close)  # Noncompliant
#       ^^^^^^^^^^^^^^^^^^^^
        return resource

    @pytest.fixture
    def yielding_fixture_in_class(self):
        resource = open_resource()
        yield resource
        resource.close()


from pytest import fixture as aliased_fixture


@aliased_fixture
def fixture_with_aliased_decorator(request):
    resource = open_resource()
    request.addfinalizer(resource.close)  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^
    return resource


def not_a_fixture_but_named_request(request):
    request.addfinalizer(cleanup)


@some_other_decorator
def decorated_but_not_fixture(request):
    request.addfinalizer(cleanup)


# Accepted FNs below — documented limitations, not expected to raise.

@pytest.fixture
def finalizer_in_local_helper(request):
    # FN: nested function skipped to avoid flagging factory-as-fixture; eager helpers are missed too.
    def register(res):
        request.addfinalizer(res.close)
    resource = open_resource()
    register(resource)
    return resource


def _register_cleanup(request, resource):
    request.addfinalizer(resource.close)


@pytest.fixture
def finalizer_via_shared_helper(request):
    # FN: addfinalizer call lives outside the fixture body.
    resource = open_resource()
    _register_cleanup(request, resource)
    return resource


@pytest.fixture
def finalizer_via_request_node(request):
    # FN: request.node.addfinalizer is a rare non-idiomatic spelling of the same API.
    resource = open_resource()
    request.node.addfinalizer(resource.close)
    return resource


import pytest_asyncio


@pytest_asyncio.fixture
async def finalizer_in_async_fixture(request):
    # FN: only pytest.fixture is recognized, not pytest_asyncio.fixture.
    resource = await open_resource()
    request.addfinalizer(resource.close)
    return resource
