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
package org.sonar.python.checks;

import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeSource;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

@Rule(key = "S5756")
public class NonCallableCalledCheck extends NonCallableCalled {

  private static final TypeMatcher EXACT_TYPE_SOURCE = TypeMatchers.hasTypeSource(TypeSource.EXACT);

  @Override
  protected boolean isException(SubscriptionContext ctx, Expression callee) {
    return !EXACT_TYPE_SOURCE.isTrueFor(callee, ctx);
  }

  @Override
  protected String message(PythonType typeV2, @Nullable String name) {
    if (name != null) {
      return "Fix this call; \"%s\"%s is not callable.".formatted(name, addTypeName(typeV2));
    }
    return "Fix this call; this expression%s is not callable.".formatted(addTypeName(typeV2));
  }
}
