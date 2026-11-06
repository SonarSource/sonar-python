import contextlib
import contextlib as ctx
from contextlib import suppress
import os
import warnings
from unittest import mock

already_bound = None
lazy import collections


def plain_import(data):
    import json  # Noncompliant {{Replace this function-level import with a module-level "lazy import".}}
#   ^^^^^^^^^^^
    return json.dumps(data)


def from_import(value):
    from decimal import Decimal  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    return Decimal(value)


def aliased_import(values):
    import statistics as stats  # Noncompliant
    return stats.mean(values)


def relative_import_to_break_a_cycle(item):
    from .users import User  # Noncompliant
    return User.get(item.owner_id)


def dotted_import(path):
    import xml.etree.ElementTree  # Noncompliant
    return xml.etree.ElementTree.parse(path)


def several_names_all_used():
    import shutil, tempfile  # Noncompliant
    return shutil.which(tempfile.gettempdir())


def import_in_a_branch(flag):
    if flag:
        import pprint  # Noncompliant
        pprint.pprint(flag)


async def async_function(data):
    import pickle  # Noncompliant
    return pickle.dumps(data)


def outer():
    def inner(data):
        import base64  # Noncompliant
        return base64.b64encode(data)
    return inner


class Service:
    def method(self, data):
        import zlib  # Noncompliant
        return zlib.compress(data)


def same_module_in_two_functions(data):
    import pathlib  # Noncompliant
    return pathlib.Path(data)


def same_module_in_another_function(data):
    import pathlib  # Noncompliant
    return pathlib.PurePath(data)


# Raised: these task decorators send only the task name, the worker imports the module itself

from airflow.decorators import task
from celery import Celery, shared_task

app = Celery()


@task
def airflow_task(value):
    import textwrap  # Noncompliant
    return textwrap.dedent(value)


@shared_task
def celery_shared_task(value):
    import fnmatch  # Noncompliant
    return fnmatch.translate(value)


@app.task
def celery_app_task(value):
    import difflib  # Noncompliant
    return difflib.SequenceMatcher(None, value, value).ratio()


# Compliant: optional dependencies, whose import errors are handled where the import happens

def optional_dependency_try():
    try:
        import tomllib
    except ImportError:
        tomllib = None
    return tomllib


def optional_dependency_try_without_rebinding():
    try:
        import tomli
        return tomli.loads("")
    except ImportError:
        return None


def optional_dependency_fallback():
    try:
        import ujson as json_impl
    except ImportError:
        import json as json_impl
    return json_impl


# Compliant: the context set up by a "with" statement would no longer apply to the import

def optional_dependency_suppress():
    with contextlib.suppress(ImportError):
        import yaml
        return yaml.safe_load("")


def optional_dependency_suppress_imported_name():
    with suppress(ImportError):
        import lxml
        return lxml.etree


def optional_dependency_suppress_module_alias():
    with ctx.suppress(ImportError):
        import orjson
        return orjson.dumps({})


def deprecated_module_with_silenced_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import imp
        return imp.get_suffixes()


def module_patched_for_the_import():
    with mock.patch.dict(os.environ, {"MPLBACKEND": "Agg"}):
        import matplotlib
        return matplotlib.get_backend()


def import_in_an_unrelated_with(path):
    with open(path) as f:
        import csv
        return list(csv.reader(f))


# Compliant: imports run for their side effects only

def side_effect_import():
    import my_project.plugins


def one_name_unused(data):
    import gzip, readline
    return gzip.compress(data)


# Compliant: moving the import would conflict with a module-level name

def name_already_bound_at_module_level():
    import already_bound
    return already_bound.value


def name_imported_eagerly_at_module_level():
    import os
    return os.getcwd()


def name_imported_lazily_at_module_level():
    import collections
    return collections.Counter()


def declared_global():
    global configparser
    import configparser
    return configparser


def same_alias_for_numpy(values):
    import numpy as xp
    return xp.asarray(values)


def same_alias_for_cupy(values):
    import cupy as xp
    return xp.asarray(values)


def same_name_from_different_modules():
    from json import loads
    return loads("{}")


def same_name_from_another_module():
    from pickle import loads
    return loads(b"")


# Compliant: a module-level binding would shadow the built-in in the whole module

def read_compressed(path):
    from gzip import open
    return open(path).read()


def py2_compatible_range(n):
    from six.moves import range
    return list(range(n))


def keep_falsy(values):
    from itertools import filterfalse as filter
    return list(filter(None, values))


# Compliant: the local name is rebound after the import

def rebound_after_import():
    import string
    string = string.ascii_letters
    return string


# Compliant: functions serialized and run in another process

from pyspark.sql.functions import udf, pandas_udf
import ray
import dask


@udf("string")
def spark_udf(value):
    import unicodedata
    return unicodedata.normalize("NFC", value)


@pandas_udf("double")
def spark_pandas_udf(series):
    import math
    return series.map(math.sqrt)


@ray.remote
def ray_task(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


@dask.delayed
def dask_task(value):
    import operator
    return operator.neg(value)


@ray.remote
class RayActor:
    def increment(self, value):
        import fractions
        return fractions.Fraction(value) + 1


@ray.remote(num_cpus=2)
class RayActorWithOptions:
    def describe(self, value):
        import locale
        return locale.str(value)


@dataclass_like
class NotAnActor:
    def method(self, value):
        import numbers  # Noncompliant
        return isinstance(value, numbers.Number)


@udf("string")
def spark_udf_with_nested_function(value):
    def helper(v):
        import html
        return html.escape(v)
    return helper(value)


@task.virtualenv(requirements=["requests"])
def airflow_virtualenv_task(value):
    import uuid
    return uuid.uuid5(uuid.NAMESPACE_URL, value)


@task.external_python(python="/opt/venv/bin/python")
def airflow_external_python_task(value):
    import secrets
    return secrets.token_hex(value)


# Compliant: not inside a function

if os.name == "nt":
    import winreg


class ClassBody:
    import numbers
    kind = numbers.Number
