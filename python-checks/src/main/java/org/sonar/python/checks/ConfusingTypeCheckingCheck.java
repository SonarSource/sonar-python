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

import java.util.List;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;

import static org.sonar.python.types.InferredTypes.containsDeclaredType;
import static org.sonar.python.types.InferredTypes.typeClassLocation;

@Rule(key = "S5864")
public class ConfusingTypeCheckingCheck extends PythonSubscriptionCheck {
  @Override
  public void initialize(Context context) {
    new NonCallableCalledCheck().initialize(context);
    new IncompatibleOperandsCheck().initialize(context);
    new ItemOperationsTypeCheck().initialize(context);
    new IterationOnNonIterableCheck().initialize(context);
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

  private static class ItemOperationsTypeCheck extends ItemOperationsType {

    @Override
    public boolean isValidSubscription(Expression subscriptionObject, String requiredMethod, @Nullable String classRequiredMethod,
                                       List<LocationInFile> secondaries) {

      InferredType type = subscriptionObject.type();
      secondaries.add(typeClassLocation(type));
      if (!InferredTypes.containsDeclaredType(type)) {
        // handled by S5644
        return true;
      }
      return type.declaresMember(requiredMethod);
    }

    @Override
    public String message(@Nullable String name, String missingMethod) {
      if (name != null) {
        return String.format("Fix this \"%s\" operation; Previous type checks suggest that \"%s\" does not have this method.", missingMethod, name);
      }
      return String.format("Fix this \"%s\" operation; Previous type checks suggest that this expression does not have this method.", missingMethod);
    }
  }

  private static class IterationOnNonIterableCheck extends IterationOnNonIterable {

    @Override
    boolean isValidIterable(Expression expression, List<LocationInFile> secondaries) {
      InferredType type = expression.type();
      secondaries.add(InferredTypes.typeClassLocation(type));
      return !containsDeclaredType(type) || type.declaresMember("__iter__") || type.declaresMember("__getitem__");
    }

    @Override
    String message(Expression expression, boolean isForLoop) {
      String typeName = InferredTypes.typeName(expression.type());
      String expressionName = nameFromExpression(expression);
      String expressionNameString = expressionName != null ? String.format("\"%s\"", expressionName) : "it";
      String typeNameString = typeName != null ? String.format("has type \"%s\" and", typeName) : "";
      return isForLoop && isAsyncIterable(expression) ?
        String.format("Add \"async\" before \"for\"; Previous type checks suggest that %s %s is an async generator.", expressionNameString, typeNameString) :
        String.format("Replace this expression; Previous type checks suggest that %s %s isn't iterable.", expressionNameString, typeNameString);
    }

    @Override
    boolean isAsyncIterable(Expression expression) {
      InferredType type = expression.type();
      return type.declaresMember("__aiter__");
    }

    private static String nameFromExpression(Expression expression) {
      if (expression.is(Tree.Kind.NAME)) {
        return ((Name) expression).name();
      }
      return null;
    }
  }
}
