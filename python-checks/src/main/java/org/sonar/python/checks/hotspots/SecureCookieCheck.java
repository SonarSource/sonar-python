/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
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

import org.sonar.check.Rule;

@Rule(key = "S2092")
public class SecureCookieCheck extends AbstractCookieFlagCheck {

  public static final String SET_COOKIE_METHOD_NAME = "set_cookie";
  public static final String SECURE_ARGUMENT_NAME = "secure";
  public static final String HEADERS_ARGUMENT_NAME = "headers";

  private final MethodArgumentsToCheckRegistry methodArgumentsToCheckRegistry = new MethodArgumentsToCheckRegistry(
    // Check for falsy secure argument
    new MethodArgumentsToCheck("django.http.response.HttpResponseBase", SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, 6),
    new MethodArgumentsToCheck("flask.wrappers.Response", SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, 6),
    new MethodArgumentsToCheck("werkzeug.wrappers.BaseResponse", SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, 6),
    new MethodArgumentsToCheck("werkzeug.sansio.response.Response", SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, 7),
    new MethodArgumentsToCheck("django.http.response.HttpResponseBase", "set_signed_cookie", SECURE_ARGUMENT_NAME, 7),
    new MethodArgumentsToCheck("fastapi.Response", SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, -1),
    new MethodArgumentsToCheck("fastapi.responses.Response", SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, -1),
    new MethodArgumentsToCheck("starlette.responses.Response", SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, -1),
    new MethodArgumentsToCheck("fastapi.responses.HTMLResponse", SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, -1),
    new MethodArgumentsToCheck("starlette.responses.HTMLResponse", SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, -1),
    new MethodArgumentsToCheck("fastapi.responses.JSONResponse", SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, -1),
    new MethodArgumentsToCheck("starlette.responses.JSONResponse", SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, -1),
    new MethodArgumentsToCheck("fastapi.responses.ORJSONResponse", SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, -1),
    new MethodArgumentsToCheck("fastapi.responses.PlainTextResponse", SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, -1),
    new MethodArgumentsToCheck("starlette.responses.PlainTextResponse", SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, -1),
    new MethodArgumentsToCheck("fastapi.responses.StreamingResponse", SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, -1),
    new MethodArgumentsToCheck("starlette.responses.StreamingResponse", SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, -1),
    new MethodArgumentsToCheck("fastapi.responses.UJSONResponse", SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, -1),
    new MethodArgumentsToCheck("fastapi.responses.FileResponse", SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, -1),
    new MethodArgumentsToCheck("starlette.responses.FileResponse", SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, -1),
    // check for insecure set-cookie header
    new MethodArgumentsToCheck("fastapi.responses.Response",  HEADERS_ARGUMENT_NAME, -1, this::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("starlette.responses.Response",  HEADERS_ARGUMENT_NAME, -1, this::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("fastapi.responses.HTMLResponse",  HEADERS_ARGUMENT_NAME, -1, this::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("starlette.responses.HTMLResponse",  HEADERS_ARGUMENT_NAME, -1, this::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("fastapi.responses.JSONResponse",  HEADERS_ARGUMENT_NAME, -1, this::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("starlette.responses.JSONResponse",  HEADERS_ARGUMENT_NAME, -1, this::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("fastapi.responses.ORJSONResponse",  HEADERS_ARGUMENT_NAME, -1, this::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("fastapi.responses.PlainTextResponse",  HEADERS_ARGUMENT_NAME, -1, this::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("starlette.responses.PlainTextResponse",  HEADERS_ARGUMENT_NAME, -1, this::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("fastapi.responses.StreamingResponse",  HEADERS_ARGUMENT_NAME, -1, this::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("starlette.responses.StreamingResponse",  HEADERS_ARGUMENT_NAME, -1, this::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("fastapi.responses.UJSONResponse",  HEADERS_ARGUMENT_NAME, -1, this::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("fastapi.responses.FileResponse",  HEADERS_ARGUMENT_NAME, -1, this::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("starlette.responses.FileResponse",  HEADERS_ARGUMENT_NAME, -1, this::isInvalidHeaderArgument)
  );

  @Override
  String flagName() {
    return SECURE_ARGUMENT_NAME;
  }

  @Override
  String message() {
    return "Make sure creating this cookie without the \"secure\" flag is safe.";
  }

  @Override
  MethodArgumentsToCheckRegistry methodArgumentsToCheckRegistry() {
    return methodArgumentsToCheckRegistry;
  }

  @Override
  protected String headerValueRegex() {
    return ".*;\\s?Secure.*";
  }
}
