from docker.utils import kwargs_from_env
from docker.utils.utils import kwargs_from_env as direct_kwargs_from_env
from docker.utils import parse_repository_tag
from google.protobuf import text_format, json_format
from google.protobuf.text_format import MessageToString
from google.protobuf.json_format import MessageToDict


def docker_versions():
    # Docker 3.7.3 and current Docker both remain supported.
    kwargs_from_env(ssl_version=2, assert_hostname=False, environment={})
    kwargs_from_env(2, False, {})
    kwargs_from_env(environment={})
    kwargs_from_env({})
    direct_kwargs_from_env(ssl_version=2, environment={})
    direct_kwargs_from_env(environment={})
    kwargs_from_env(nonexistent=True)  # Noncompliant {{Remove this unexpected named argument 'nonexistent'.}}
    direct_kwargs_from_env(nonexistent=True)  # Noncompliant {{Remove this unexpected named argument 'nonexistent'.}}
    parse_repository_tag(nonexistent=True)  # Noncompliant {{Add 1 missing arguments; 'parse_repository_tag' expects 1 positional arguments.}} {{Remove this unexpected named argument 'nonexistent'.}}


def protobuf_7_only(message):
    # Only Protobuf 7.x is modeled by the upgraded Typeshed stubs. The findings
    # below are accepted FPs for projects using older dependencies, not API bugs
    # in those projects. Revisit compatibility if user reports justify it.
    # Protobuf 6.x still accepts these float options; Protobuf 7.x removed them.
    text_format.MessageToString(message, float_format=".3g", double_format=".15g")  # Noncompliant {{Remove this unexpected named argument 'float_format'.}} {{Remove this unexpected named argument 'double_format'.}}
    MessageToString(message, float_format=".3g")  # Noncompliant {{Remove this unexpected named argument 'float_format'.}}
    json_format.MessageToJson(message, float_precision=3)  # Noncompliant {{Remove this unexpected named argument 'float_precision'.}}
    MessageToDict(message, always_print_fields_with_no_presence=True, float_precision=3)  # Noncompliant {{Remove this unexpected named argument 'float_precision'.}}
    # Protobuf 4.x accepts this option; Protobuf 5.x and later replaced it.
    json_format.MessageToDict(message, including_default_value_fields=True)  # Noncompliant {{Remove this unexpected named argument 'including_default_value_fields'.}}
    MessageToDict(message, including_default_value_fields=True)  # Noncompliant {{Remove this unexpected named argument 'including_default_value_fields'.}}

    # Supported Protobuf 7.x calls and genuine invalid-call controls.
    text_format.MessageToString(message, force_colon=True)
    MessageToString(message, force_colon=True)
    json_format.MessageToDict(message, always_print_fields_with_no_presence=True, unquote_int64_if_possible=True)
    MessageToDict(message, always_print_fields_with_no_presence=True)
    text_format.MessageToString(message, nonexistent=True)  # Noncompliant {{Remove this unexpected named argument 'nonexistent'.}}
    MessageToString(message, nonexistent=True)  # Noncompliant {{Remove this unexpected named argument 'nonexistent'.}}
    json_format.MessageToDict(message, nonexistent=True)  # Noncompliant {{Remove this unexpected named argument 'nonexistent'.}}
    MessageToDict(message, nonexistent=True)  # Noncompliant {{Remove this unexpected named argument 'nonexistent'.}}
    text_format.MessageToString()  # Noncompliant {{Add 1 missing arguments; 'MessageToString' expects at least 1 positional arguments.}}
    json_format.MessageToDict()  # Noncompliant {{Add 1 missing arguments; 'MessageToDict' expects at least 1 positional arguments.}}
    # Other declarations in the upstream modules still resolve.
    text_format.TextWriter(True).write()  # Noncompliant {{Add 1 missing arguments; 'write' expects 1 positional arguments.}}
    json_format.Parse("{}")  # Noncompliant {{Add 1 missing arguments; 'Parse' expects at least 2 positional arguments.}}
