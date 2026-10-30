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
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.UnknownType;

public class CanReturnTypeSatisfyingPredicate implements TypePredicate {
  private final TypePredicate returnTypePredicate;

  public CanReturnTypeSatisfyingPredicate(TypePredicate returnTypePredicate) {
    this.returnTypePredicate = returnTypePredicate;
  }

  @Override
  public TriBool check(PythonType type, TypePredicateContext ctx) {
    // A callable alternative that can return a matching value is sufficient.
    return evaluateAny(this::checkCallable, type, ctx);
  }

  private TriBool checkCallable(PythonType type, TypePredicateContext ctx) {
    if (!(type instanceof FunctionType function) || function.hasDecorators() || function.isAsynchronous()) {
      return TriBool.UNKNOWN;
    }
    return evaluateAny(returnTypePredicate, function.returnType(), ctx);
  }

  private static TriBool evaluateAny(TypePredicate predicate, PythonType type, TypePredicateContext ctx) {
    if (type instanceof UnknownType) {
      return TriBool.UNKNOWN;
    }
    if (type instanceof UnionType union) {
      return union.candidates().stream()
        .map(candidate -> evaluateAny(predicate, candidate, ctx))
        .reduce(TriBool.FALSE, TriBool::or);
    }
    if (type instanceof ObjectType object && object.unwrappedType() instanceof UnionType union) {
      return union.candidates().stream()
        .map(candidate -> evaluateAny(predicate, ObjectType.fromType(candidate), ctx))
        .reduce(TriBool.FALSE, TriBool::or);
    }
    return TypePredicateUtils.evaluate(predicate, type, ctx);
  }
}
