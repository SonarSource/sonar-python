/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks.hotspots;

import java.util.List;
import java.util.Set;

final class CookieApi {

  static final String SET_COOKIE_METHOD_NAME = "set_cookie";
  static final String SET_SIGNED_COOKIE_METHOD_NAME = "set_signed_cookie";

  static final String DJANGO_RESPONSE = "django.http.response.HttpResponseBase";
  static final String FLASK_RESPONSE = "flask.wrappers.Response";
  static final String WERKZEUG_BASE_RESPONSE = "werkzeug.wrappers.BaseResponse";
  static final String WERKZEUG_RESPONSE = "werkzeug.sansio.response.Response";
  static final String STARLETTE_RESPONSE = "starlette.responses.Response";
  static final String FASTAPI_RESPONSE = "fastapi.Response";

  // Base types for checks that recognize response instances through inheritance.
  static final Set<String> RESPONSE_BASE_TYPES = Set.of(
    FLASK_RESPONSE, WERKZEUG_BASE_RESPONSE, WERKZEUG_RESPONSE, DJANGO_RESPONSE, STARLETTE_RESPONSE);

  // Explicit aliases and subclasses used by the cookie-flag method registries.
  static final List<String> ASGI_RESPONSE_TYPES = List.of(
    FASTAPI_RESPONSE,
    "fastapi.responses.Response",
    STARLETTE_RESPONSE,
    "fastapi.responses.HTMLResponse",
    "starlette.responses.HTMLResponse",
    "fastapi.responses.JSONResponse",
    "starlette.responses.JSONResponse",
    "fastapi.responses.ORJSONResponse",
    "fastapi.responses.PlainTextResponse",
    "starlette.responses.PlainTextResponse",
    "fastapi.responses.StreamingResponse",
    "starlette.responses.StreamingResponse",
    "fastapi.responses.UJSONResponse",
    "fastapi.responses.FileResponse",
    "starlette.responses.FileResponse");

  private CookieApi() {
  }
}
