import os
import sys

import myapp.client
import myapp.config
import pytest


def test_api_key():
    old_key = os.environ.get('API_KEY')
    os.environ['API_KEY'] = 'test_key'  # Noncompliant {{Use the "monkeypatch" fixture for temporary modifications instead of manually modifying global state.}}
#   ^^^^^^^^^^^^^^^^^^^^^
    result = get_api_configuration()
    assert result['key'] == 'test_key'
    if old_key:
        os.environ['API_KEY'] = old_key  # Noncompliant {{Use the "monkeypatch" fixture for temporary modifications instead of manually modifying global state.}}
    else:
        del os.environ['API_KEY']  # Noncompliant {{Use the "monkeypatch" fixture for temporary modifications instead of manually modifying global state.}}


def test_api_key_compliant(monkeypatch):
    monkeypatch.setenv('API_KEY', 'test_key')
    result = get_api_configuration()
    assert result['key'] == 'test_key'


def test_feature_flag():
    old_value = myapp.config.ENABLE_FEATURE
    myapp.config.ENABLE_FEATURE = True  # Noncompliant {{Use the "monkeypatch" fixture for temporary modifications instead of manually modifying global state.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    assert myapp.is_feature_enabled()
    myapp.config.ENABLE_FEATURE = old_value  # Noncompliant {{Use the "monkeypatch" fixture for temporary modifications instead of manually modifying global state.}}


def test_feature_flag_compliant(monkeypatch):
    monkeypatch.setattr(myapp.config, 'ENABLE_FEATURE', True)
    assert myapp.is_feature_enabled()


def test_notify_user():
    client = myapp.client.Client()
    original_send = client.send
    client.send = lambda message: {"sent": True}
    result = client.notify("Hello")
    assert result["sent"]
    client.send = original_send


def test_notify_user_compliant(monkeypatch):
    client = myapp.client.Client()
    monkeypatch.setattr(client, "send", lambda message: {"sent": True})
    result = client.notify("Hello")
    assert result["sent"]


def helper_set_env():
    os.environ['API_KEY'] = 'test_key'


def test_calls_helper():
    helper_set_env()
    assert True


def not_a_test():
    os.environ['API_KEY'] = 'test_key'


def test_nested_helper():
    def inner():
        os.environ['API_KEY'] = 'test_key'

    inner()
    assert True


def test_parenthesized_environ():
    (os.environ)['API_KEY'] = 'test_key'  # Noncompliant {{Use the "monkeypatch" fixture for temporary modifications instead of manually modifying global state.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^


def test_multiple_delete():
    os.environ['A'] = '1'  # Noncompliant {{Use the "monkeypatch" fixture for temporary modifications instead of manually modifying global state.}}
    os.environ['B'] = '2'  # Noncompliant {{Use the "monkeypatch" fixture for temporary modifications instead of manually modifying global state.}}
    del os.environ['A']  # Noncompliant {{Use the "monkeypatch" fixture for temporary modifications instead of manually modifying global state.}}


@pytest.fixture
def test_with_fixture_decorator():
    os.environ['FIXTURE_VAR'] = '1'


@pytest.fixture()
def test_with_call_fixture_decorator():
    os.environ['FIXTURE_VAR'] = '1'


@pytest.fixture
def setup_env():
    os.environ['API_KEY'] = 'test_key'
    yield
    del os.environ['API_KEY']


@pytest.fixture()
def setup_env_with_call_decorator():
    os.environ['TEMP_VAR'] = 'temp'
    yield
    del os.environ['TEMP_VAR']


def setup_module():
    os.environ['API_KEY'] = 'test_key'


def teardown_module():
    del os.environ['API_KEY']


class TestApp:
    def test_instance_attribute(self):
        self.value = 1
        assert self.value == 1

    def setup_method(self):
        os.environ['SETUP'] = '1'

    def teardown_method(self):
        del os.environ['SETUP']

def test_sys_path_insert():
    sys.path.insert(0, '/tmp/test')  # Noncompliant {{Use the "monkeypatch" fixture for temporary modifications instead of manually modifying global state.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def test_sys_path_subscription():
    sys.path[0] = '/tmp/test'  # Noncompliant {{Use the "monkeypatch" fixture for temporary modifications instead of manually modifying global state.}}
#   ^^^^^^^^^^^


def test_module_attribute_subscription_chain():
    myapp.config.SETTINGS['key'].value = 'value'  # Noncompliant {{Use the "monkeypatch" fixture for temporary modifications instead of manually modifying global state.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def test_sys_path_insert_compliant(monkeypatch):
    monkeypatch.syspath_prepend('/tmp/test')


# --- FP analysis reproducers (should remain compliant) ---

class PodSpec:
    def __init__(self):
        self.containers = [type('C', (), {})()]


class Pod:
    def __init__(self):
        self.spec = PodSpec()
        self.metadata = type('M', (), {'labels': {}, 'annotations': None, 'name': None, 'namespace': None})()


def test_local_nested_subscription_chain():
    expected = Pod()
    expected.metadata.labels = {}
    expected.metadata.name = 'pod_id'
    expected.metadata.namespace = 'test_namespace'
    expected.spec.containers[0].args = ['command']
    expected.spec.containers[0].image = 'img'
    expected.spec.containers[0].resources = {'limits': {'cpu': '1m', 'memory': '1G'}}


def test_local_subscription_then_attribute():
    clf = type('Clf', (), {'inputs': [type('In', (), {})()], 'outputs': [type('Out', (), {'in_blocks': [type('B', (), {})()]})()]})()
    clf.inputs[0].shape = (32, 32, 3)
    clf.outputs[0].in_blocks[0].shape = (10,)


def test_mock_call_return_attribute_assignment():
    mock_post = type('M', (), {'__call__': lambda self: type('R', (), {})()})()
    mock_post().ok = False
    mock_post().text = 'content'
    mock_post().headers = {}


def test_call_return_attribute_assignment():
    def get_ti_from_db(task):
        return type('TI', (), {'state': None})()

    get_ti_from_db('task_1').state = 'FAILED'
    get_ti_from_db('task_2').state = 'SUCCESS'


def test_getattr_result_attribute_assignment():
    float_frame = type('F', (), {'index': type('I', (), {'name': None})()})()
    getattr(float_frame, 'index').name = 'foo'


def test_nested_self_subscription_attribute():
    class TestBuilders:
        def __init__(self):
            self.builders = {'A': type('B', (), {})()}

        def test_nested_self(self):
            self.builders['A'].maybeStartBuild = lambda: None


def test_workbook_local_nested_assignment():
    wb = type('WB', (), {'worksheets': [type('WS', (), {'title': None})(), type('WS', (), {})()]})()
    wb.worksheets[0].title = 'foo'
    wb.worksheets[0]['A1'] = type('Cell', (), {'value': None})()
    wb.worksheets[0]['A1'].value = 'foo'


def test_local_dataframe_nested_flags():
    df = type('DF', (), {'_mgr': type('Mgr', (), {'blocks': [type('Blk', (), {'values': type('V', (), {'flags': type('F', (), {'writeable': True})()})()})()]})()})()
    df._mgr.blocks[0].values.flags.writeable = False


def test_module_attr_still_detected():
    myapp.config.ENABLE_FEATURE = True  # Noncompliant {{Use the "monkeypatch" fixture for temporary modifications instead of manually modifying global state.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
