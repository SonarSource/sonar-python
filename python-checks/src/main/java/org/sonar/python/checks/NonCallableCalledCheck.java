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
package org.sonar.python.checks;

import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;

@Rule(key = "S5756")
public class NonCallableCalledCheck extends NonCallableCalled {

  @Override
  protected boolean isExpectedTypeSource(SubscriptionContext ctx, PythonType calleeType) {
    return ctx.typeChecker().typeCheckBuilder().isExactTypeSource().check(calleeType) == TriBool.TRUE;
  }

  @Override
  protected String message(PythonType typeV2, @Nullable String name) {
    if (name != null) {
      return "Fix this call; \"%s\"%s is not callable.".formatted(name, addTypeName(typeV2));
    }
    return "Fix this call; this expression%s is not callable.".formatted(addTypeName(typeV2));
  }
}
