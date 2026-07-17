import os


def setup_module():
    os.environ['MODULE_SETUP'] = '1'


def teardown_module():
    del os.environ['MODULE_SETUP']


def setup_class():
    os.environ['CLASS_SETUP'] = '1'


def teardown_class():
    del os.environ['CLASS_SETUP']


def setup_method():
    os.environ['METHOD_SETUP'] = '1'


def teardown_method():
    del os.environ['METHOD_SETUP']


class TestLifecycle:
    def setUp(self):
        os.environ['SETUP'] = '1'

    def tearDown(self):
        del os.environ['SETUP']

    def setUpClass(cls):
        os.environ['SETUP_CLASS'] = '1'

    def tearDownClass(cls):
        del os.environ['SETUP_CLASS']

    def test_ok(self):
        assert True
