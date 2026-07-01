def helper_only_module():  # Noncompliant@-1 {{Add some tests to this file.}}
    return 42


class TestHelperOnly:
    def helper(self):
        return 42
