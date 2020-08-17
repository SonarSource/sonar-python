/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.python.checks;

import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;

import static org.sonar.python.types.InferredTypes.containsDeclaredType;

@Rule(key = "S5864")
public class ConfusingTypeCheckingCheck extends PythonSubscriptionCheck {
  @Override
  public void initialize(Context context) {
    new NonCallableCalledCheck().initialize(context);
    new IncompatibleOperandsCheck().initialize(context);
  }

  private static class NonCallableCalledCheck extends NonCallableCalled {

    @Override
    public boolean isNonCallableType(InferredType type) {
      return containsDeclaredType(type) && !type.declaresMember("__call__");
    }

    @Override
    public String message(InferredType calleeType, @Nullable String name) {
      if (name != null) {
        return String.format("Fix this call; Previous type checks suggest that \"%s\"%s is not callable.", name, addTypeName(calleeType));
      }
      return String.format("Fix this call; Previous type checks suggest that this expression%s is not callable.", addTypeName(calleeType));
    }
  }

  private static class IncompatibleOperandsCheck extends IncompatibleOperands {
    @Override
    public SpecialMethod resolveMethod(InferredType type, String method) {
      Symbol symbol = type.resolveDeclaredMember(method).orElse(null);
      boolean isUnresolved = !containsDeclaredType(type) || (symbol == null && type.declaresMember(method));
      return new SpecialMethod(symbol, isUnresolved);
    }

    @Override
    public String message(Token operator) {
      return "Fix this \"" + operator.value() + "\" operation; Previous type checks suggest that operand has incompatible type.";
    }

    @Override
    public String message(Token operator, InferredType left, InferredType right) {
      String leftTypeName = InferredTypes.typeName(left);
      String rightTypeName = InferredTypes.typeName(right);
      String message = "Fix this \"" + operator.value() + "\" operation; Previous type checks suggest that operands have incompatible types";
      if (leftTypeName != null && rightTypeName != null) {
        message += " (" + leftTypeName + " and " + rightTypeName + ")";
      }
      return message + ".";
    }
  }
}
