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

import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;

import static org.sonar.plugins.python.api.types.BuiltinTypes.NONE_TYPE;

@Rule(key = "S2159")
public class SillyEqualityCheck extends SillyEquality {

  @Override
  boolean areIdentityComparableOrNone(InferredType leftType, InferredType rightType) {
    return leftType.isIdentityComparableWith(rightType) || leftType.canOnlyBe(NONE_TYPE) || rightType.canOnlyBe(NONE_TYPE);
  }

  @Override
  public boolean canImplementEqOrNe(Expression expression) {
    InferredType type = expression.type();
    return type.canHaveMember("__eq__") || type.canHaveMember("__ne__");
  }

  @CheckForNull
  @Override
  String builtinTypeCategory(InferredType inferredType) {
    return InferredTypes.getBuiltinCategory(inferredType);
  }

  @Override
  String message(String result) {
    return String.format("Remove this equality check between incompatible types; it will always return %s.", result);
  }
}
