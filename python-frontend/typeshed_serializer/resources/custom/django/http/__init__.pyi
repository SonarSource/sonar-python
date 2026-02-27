import django.http.request as request
import django.http.response as response

from .request import (
    HttpRequest as HttpRequest,
)

from .response import (
    HttpResponse as HttpResponse,
    HttpResponseBadRequest as HttpResponseBadRequest,
    HttpResponseForbidden as HttpResponseForbidden,
    HttpResponseGone as HttpResponseGone,
    HttpResponseNotAllowed as HttpResponseNotAllowed,
    HttpResponseNotFound as HttpResponseNotFound,
    HttpResponseNotModified as HttpResponseNotModified,
    HttpResponsePermanentRedirect as HttpResponsePermanentRedirect,
    HttpResponseRedirect as HttpResponseRedirect,
    HttpResponseServerError as HttpResponseServerError,
)
