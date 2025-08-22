# this custom stubs redefines HTTPStatus and HTTPMethod, but type-hints the fields "wrongly" as a HTTPStatus or HTTPMethod respectively. 
# This causes the field type to be correctly serialized as a HTTPStatus or HTTPMethod instead of an int

import sys
from enum import IntEnum

if sys.version_info >= (3, 11):
    from enum import StrEnum

if sys.version_info >= (3, 11):
    __all__ = ["HTTPStatus", "HTTPMethod"]
else:
    __all__ = ["HTTPStatus"]

class HTTPStatus(IntEnum):
    @property
    def phrase(self) -> str: ...
    @property
    def description(self) -> str: ...

    CONTINUE: "HTTPStatus"
    SWITCHING_PROTOCOLS: "HTTPStatus"
    PROCESSING: "HTTPStatus"
    OK: "HTTPStatus"
    CREATED: "HTTPStatus"
    ACCEPTED: "HTTPStatus"
    NON_AUTHORITATIVE_INFORMATION: "HTTPStatus"
    NO_CONTENT: "HTTPStatus"
    RESET_CONTENT: "HTTPStatus"
    PARTIAL_CONTENT: "HTTPStatus"
    MULTI_STATUS: "HTTPStatus"
    ALREADY_REPORTED: "HTTPStatus"
    IM_USED: "HTTPStatus"
    MULTIPLE_CHOICES: "HTTPStatus"
    MOVED_PERMANENTLY: "HTTPStatus"
    FOUND: "HTTPStatus"
    SEE_OTHER: "HTTPStatus"
    NOT_MODIFIED: "HTTPStatus"
    USE_PROXY: "HTTPStatus"
    TEMPORARY_REDIRECT: "HTTPStatus"
    PERMANENT_REDIRECT: "HTTPStatus"
    BAD_REQUEST: "HTTPStatus"
    UNAUTHORIZED: "HTTPStatus"
    PAYMENT_REQUIRED: "HTTPStatus"
    FORBIDDEN: "HTTPStatus"
    NOT_FOUND: "HTTPStatus"
    METHOD_NOT_ALLOWED: "HTTPStatus"
    NOT_ACCEPTABLE: "HTTPStatus"
    PROXY_AUTHENTICATION_REQUIRED: "HTTPStatus"
    REQUEST_TIMEOUT: "HTTPStatus"
    CONFLICT: "HTTPStatus"
    GONE: "HTTPStatus"
    LENGTH_REQUIRED: "HTTPStatus"
    PRECONDITION_FAILED: "HTTPStatus"
    REQUEST_ENTITY_TOO_LARGE: "HTTPStatus"
    REQUEST_URI_TOO_LONG: "HTTPStatus"
    UNSUPPORTED_MEDIA_TYPE: "HTTPStatus"
    REQUESTED_RANGE_NOT_SATISFIABLE: "HTTPStatus"
    EXPECTATION_FAILED: "HTTPStatus"
    UNPROCESSABLE_ENTITY: "HTTPStatus"
    LOCKED: "HTTPStatus"
    FAILED_DEPENDENCY: "HTTPStatus"
    UPGRADE_REQUIRED: "HTTPStatus"
    PRECONDITION_REQUIRED: "HTTPStatus"
    TOO_MANY_REQUESTS: "HTTPStatus"
    REQUEST_HEADER_FIELDS_TOO_LARGE: "HTTPStatus"
    INTERNAL_SERVER_ERROR: "HTTPStatus"
    NOT_IMPLEMENTED: "HTTPStatus"
    BAD_GATEWAY: "HTTPStatus"
    SERVICE_UNAVAILABLE: "HTTPStatus"
    GATEWAY_TIMEOUT: "HTTPStatus"
    HTTP_VERSION_NOT_SUPPORTED: "HTTPStatus"
    VARIANT_ALSO_NEGOTIATES: "HTTPStatus"
    INSUFFICIENT_STORAGE: "HTTPStatus"
    LOOP_DETECTED: "HTTPStatus"
    NOT_EXTENDED: "HTTPStatus"
    NETWORK_AUTHENTICATION_REQUIRED: "HTTPStatus"
    MISDIRECTED_REQUEST: "HTTPStatus"
    if sys.version_info >= (3, 8):
        UNAVAILABLE_FOR_LEGAL_REASONS: "HTTPStatus"
    if sys.version_info >= (3, 9):
        EARLY_HINTS: "HTTPStatus"  # Literal[103]
        IM_A_TEAPOT: "HTTPStatus"  # Literal[418]
        TOO_EARLY: "HTTPStatus"  # Literal[425]

if sys.version_info >= (3, 11):
    class HTTPMethod(StrEnum):
        @property
        def description(self) -> str: ...
        CONNECT: "HTTPMethod"
        DELETE: "HTTPMethod"
        GET: "HTTPMethod"
        HEAD: "HTTPMethod"
        OPTIONS: "HTTPMethod"
        PATCH: "HTTPMethod"
        POST: "HTTPMethod"
        PUT: "HTTPMethod"
        TRACE: "HTTPMethod"
