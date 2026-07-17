import pytest


def helper(request):
    request.getfixturevalue('database')
