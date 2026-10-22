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
package org.sonar.python.types.v2.matchers;

import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnknownType;

/**
 * Matches functions with at least one decorator, all satisfying the supplied predicate.
 * Unknown decorator entries must be preserved by producers of the function type.
 * A definite mismatch takes precedence over unknown entries.
 */
public class HasOnlyFunctionDecoratorsSatisfyingPredicate implements TypePredicate {
  private final TypePredicate decoratorPredicate;

  public HasOnlyFunctionDecoratorsSatisfyingPredicate(TypePredicate decoratorPredicate) {
    this.decoratorPredicate = decoratorPredicate;
  }

  @Override
  public TriBool check(PythonType type, TypePredicateContext ctx) {
    if (type instanceof UnknownType) {
      return TriBool.UNKNOWN;
    }
    if (!(type instanceof FunctionType function) || function.decorators().isEmpty()) {
      return TriBool.FALSE;
    }
    TriBool result = TriBool.TRUE;
    for (var decorator : function.decorators()) {
      result = result.logicalAnd(TypePredicateUtils.evaluate(decoratorPredicate, decorator.type(), ctx));
      if (result == TriBool.FALSE) {
        return TriBool.FALSE;
      }
    }
    return result;
  }
}
