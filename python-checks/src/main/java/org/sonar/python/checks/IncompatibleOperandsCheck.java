/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;

@Rule(key = "S5607")
public class IncompatibleOperandsCheck extends IncompatibleOperands {

  @Override
  public SpecialMethod resolveMethod(InferredType type, String method) {
    Symbol resolvedMethod = type.resolveMember(method).orElse(null);
    boolean isUnresolved = resolvedMethod == null && type.canHaveMember(method);
    return new SpecialMethod(resolvedMethod, isUnresolved);
  }

  @Override
  public String message(Token operator, InferredType left, InferredType right) {
    String leftTypeName = InferredTypes.typeName(left);
    String rightTypeName = InferredTypes.typeName(right);
    String message = "Fix this invalid \"" + operator.value() + "\" operation between incompatible types";
    if (leftTypeName != null && rightTypeName != null) {
      message += " (" + leftTypeName + " and " + rightTypeName + ")";
    }
    return message + ".";
  }

  @Override
  public String message(Token operator) {
    return "Fix this invalid \"" + operator.value() + "\" operation on a type which doesn't support it.";
  }
}
