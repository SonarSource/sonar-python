import logging
import os
import socket
import stripe
from stripe import StripeObject
from os import urandom as random_bytes
from pathlib import Path
from logging import getLogger
from sqlalchemy import and_, or_
from sqlalchemy.sql.elements import BooleanClauseList, ColumnElement
from typing import Optional, Union


class CustomLogger(logging.Logger):
    def custom_method(self):
        return 42


def logger_factory() -> CustomLogger:
    logging.setLoggerClass(CustomLogger)
    return logging.getLogger(__name__)


def parenthesized_factory() -> CustomLogger:
    return (logging.getLogger(__name__))


class Unrelated:
    def missing_method(self):
        return 42


def unrelated_annotation() -> Unrelated:
    return logging.getLogger(__name__)  # Noncompliant


def unrelated_builtin() -> int:
    return logging.getLogger(__name__)  # Noncompliant


class Base:
    pass


class Child(Base):
    def child_method(self):
        return 42


def exact_constructor() -> Child:
    return Base()  # Noncompliant


def exact_library_constructor() -> CustomLogger:
    return logging.Logger("name")  # Noncompliant


def aliased_factory() -> CustomLogger:
    get_logger = logging.getLogger
    return get_logger(__name__)


class GenericLogger[T](logging.Logger):
    def generic_method(self):
        return 42


def parameterized_annotation() -> GenericLogger[int]:
    return logging.getLogger(__name__)  # Noncompliant


def optional_factory() -> Optional[CustomLogger]:
    return logging.getLogger(__name__)


def union_factory() -> Union[int, CustomLogger]:
    return logging.getLogger(__name__)


def incompatible_union() -> Union[int, Unrelated]:
    return logging.getLogger(__name__)  # Noncompliant


def optional_generic() -> Optional[GenericLogger[int]]:
    return logging.getLogger(__name__)  # Noncompliant


def modern_union_factory() -> CustomLogger | None:
    return logging.getLogger(__name__)


def incompatible_modern_union() -> int | Unrelated:
    return logging.getLogger(__name__)  # Noncompliant


def right_union_factory() -> int | CustomLogger:
    return logging.getLogger(__name__)


def bare_generic_annotation() -> GenericLogger:
    return logging.getLogger(__name__)  # Noncompliant


# Documents a legacy inference limitation; this does not exercise the stub-origin guard.
def local_logger():
    return logging.Logger("name")


def local_factory() -> CustomLogger:
    return local_logger()  # FN: legacy inference does not propagate the local function's return type.


def imported_factory() -> CustomLogger:
    return getLogger("name")


class SecureBytes(bytes):
    def secure(self):
        pass


def exact_stub_result() -> SecureBytes:
    return os.urandom(8)  # Noncompliant


def aliased_exact_stub_result() -> Optional[SecureBytes]:
    return (random_bytes(8))  # Noncompliant


def file_contents() -> SecureBytes:
    return Path("input").read_bytes()  # Noncompliant


class UserId(int):
    def validate_id(self):
        pass


def user_id() -> UserId:
    return len("hello")  # Noncompliant


class UserName(str):
    def validate_name(self):
        pass


def user_name() -> UserName:
    return "HELLO".lower()  # Noncompliant


class CustomSocket(socket.socket):
    def custom_method(self):
        pass


def exact_library_result() -> CustomSocket:
    return socket.create_connection(("localhost", 80))  # Noncompliant


def incompatible_library_result() -> SecureBytes:
    return socket.create_connection(("localhost", 80))  # Noncompliant


def externally_configured_logger(name) -> CustomLogger:
    # The logger class can be configured in another module.
    factory = getLogger
    return (factory(name))


def child_logger(logger: logging.Logger) -> CustomLogger:
    return logger.getChild("child")


# The upstream ColumnElement[bool] return annotations permit these subclasses.
def conjunction(left, right) -> BooleanClauseList:
    return and_(left, right)


def disjunction(left, right) -> BooleanClauseList:
    return or_(left, right)


def aliased_conjunction(clauses) -> Optional[BooleanClauseList]:
    combine = and_
    return (combine(*clauses))


class CustomExpression(ColumnElement):
    def custom_method(self):
        return 42


def preserved_conjunction(expression: CustomExpression) -> CustomExpression:
    return and_(expression)


def preserved_disjunction(expression: CustomExpression) -> CustomExpression | None:
    return or_(expression)


# Stripe uses its current inline typing after the Typeshed upgrade.


def subscribe_webhook() -> Optional[StripeObject]:
    return stripe.WebhookEndpoint.create(url="https://example.com/webhook")


def retrieve_customer() -> Optional[StripeObject]:
    return stripe.Customer.retrieve("cus_test")


def create_customer() -> Optional[StripeObject]:
    return stripe.Customer.create(email="test@example.com")
