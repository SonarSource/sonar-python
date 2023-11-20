/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.checks.hotspots;

import java.util.Optional;
import java.util.function.Predicate;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.python.checks.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S2092")
public class SecureCookieCheck extends AbstractCookieFlagCheck {

  public static final String SET_COOKIE_METHOD_NAME = "set_cookie";
  public static final String SECURE_ARGUMENT_NAME = "secure";
  public static final String HEADERS_ARGUMENT_NAME = "headers";
  private static final MethodArgumentsToCheckRegistry METHOD_ARGUMENTS_TO_CHECK_REGISTRY = new MethodArgumentsToCheckRegistry(
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
    new MethodArgumentsToCheck("fastapi.responses.Response",  HEADERS_ARGUMENT_NAME, -1, SecureCookieCheck::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("starlette.responses.Response",  HEADERS_ARGUMENT_NAME, -1, SecureCookieCheck::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("fastapi.responses.HTMLResponse",  HEADERS_ARGUMENT_NAME, -1, SecureCookieCheck::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("starlette.responses.HTMLResponse",  HEADERS_ARGUMENT_NAME, -1, SecureCookieCheck::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("fastapi.responses.JSONResponse",  HEADERS_ARGUMENT_NAME, -1, SecureCookieCheck::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("starlette.responses.JSONResponse",  HEADERS_ARGUMENT_NAME, -1, SecureCookieCheck::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("fastapi.responses.ORJSONResponse",  HEADERS_ARGUMENT_NAME, -1, SecureCookieCheck::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("fastapi.responses.PlainTextResponse",  HEADERS_ARGUMENT_NAME, -1, SecureCookieCheck::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("starlette.responses.PlainTextResponse",  HEADERS_ARGUMENT_NAME, -1, SecureCookieCheck::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("fastapi.responses.StreamingResponse",  HEADERS_ARGUMENT_NAME, -1, SecureCookieCheck::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("starlette.responses.StreamingResponse",  HEADERS_ARGUMENT_NAME, -1, SecureCookieCheck::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("fastapi.responses.UJSONResponse",  HEADERS_ARGUMENT_NAME, -1, SecureCookieCheck::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("fastapi.responses.FileResponse",  HEADERS_ARGUMENT_NAME, -1, SecureCookieCheck::isInvalidHeaderArgument),
    new MethodArgumentsToCheck("starlette.responses.FileResponse",  HEADERS_ARGUMENT_NAME, -1, SecureCookieCheck::isInvalidHeaderArgument)
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
    return METHOD_ARGUMENTS_TO_CHECK_REGISTRY;
  }

  private static boolean isInvalidHeaderArgument(@Nullable RegularArgument argument) {
    return Optional.ofNullable(argument)
      .map(RegularArgument::expression)
      .map(SecureCookieCheck::isDictWithSensitiveEntry)
      .orElse(false);
  }

  private static boolean isDictWithSensitiveEntry(Expression expression) {
    return TreeUtils.toOptionalInstanceOf(Name.class, expression)
      .map(Expressions::singleAssignedNonNameValue)
      .map(SecureCookieCheck::isDictWithSensitiveEntry)
      .or(() -> TreeUtils.toOptionalInstanceOf(DictionaryLiteral.class, expression)
        .map(SecureCookieCheck::hasTrueVerifySignatureEntry)
      ).orElse(false);
  }

  private static boolean hasTrueVerifySignatureEntry(DictionaryLiteral dictionaryLiteral) {
    return dictionaryLiteral.elements().stream()
      .filter(KeyValuePair.class::isInstance)
      .map(KeyValuePair.class::cast)
      .filter(keyValuePair -> isSensitiveKey(keyValuePair.key()))
      .map(KeyValuePair::value)
      .anyMatch(SecureCookieCheck::invalidSetCookieHeaderValue);
  }

  private static boolean isSensitiveKey(Expression key) {
    return TreeUtils.toOptionalInstanceOf(StringLiteral.class, key)
      .map(StringLiteral::trimmedQuotesValue)
      .filter("set-cookie"::equalsIgnoreCase)
      .isPresent();
  }

  private static boolean invalidSetCookieHeaderValue(Expression value) {
    return TreeUtils.toOptionalInstanceOf(StringLiteral.class, value)
      .map(StringLiteral::trimmedQuotesValue)
      .filter(Predicate.not(val -> val.matches(".*;\\s?Secure")))
      .isPresent();
  }
}
