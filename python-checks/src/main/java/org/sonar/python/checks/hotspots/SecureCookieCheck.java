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

import java.util.ArrayList;
import java.util.List;
import org.sonar.check.Rule;

@Rule(key = "S2092")
public class SecureCookieCheck extends AbstractCookieFlagCheck {

  public static final String SECURE_ARGUMENT_NAME = "secure";
  public static final String HEADERS_ARGUMENT_NAME = "headers";

  private final MethodArgumentsToCheckRegistry methodArgumentsToCheckRegistry = createMethodArgumentsToCheckRegistry();

  private MethodArgumentsToCheckRegistry createMethodArgumentsToCheckRegistry() {
    var methods = new ArrayList<>(List.of(
      new MethodArgumentsToCheck(CookieApi.DJANGO_RESPONSE, CookieApi.SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, 6),
      new MethodArgumentsToCheck(CookieApi.FLASK_RESPONSE, CookieApi.SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, 6),
      new MethodArgumentsToCheck(CookieApi.WERKZEUG_BASE_RESPONSE, CookieApi.SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, 6),
      new MethodArgumentsToCheck(CookieApi.WERKZEUG_RESPONSE, CookieApi.SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, 7),
      new MethodArgumentsToCheck(CookieApi.DJANGO_RESPONSE, CookieApi.SET_SIGNED_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, 7)));
    for (String responseType : CookieApi.ASGI_RESPONSE_TYPES) {
      methods.add(new MethodArgumentsToCheck(responseType, CookieApi.SET_COOKIE_METHOD_NAME, SECURE_ARGUMENT_NAME, -1));
      // Constructor-header support excludes the top-level FastAPI alias.
      if (!CookieApi.FASTAPI_RESPONSE.equals(responseType)) {
        methods.add(new MethodArgumentsToCheck(responseType, HEADERS_ARGUMENT_NAME, -1, this::isInvalidHeaderArgument));
      }
    }
    return new MethodArgumentsToCheckRegistry(methods.toArray(MethodArgumentsToCheck[]::new));
  }

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
