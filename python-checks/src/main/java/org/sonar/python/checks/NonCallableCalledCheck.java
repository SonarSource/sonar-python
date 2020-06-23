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

import java.util.Arrays;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;

import static org.sonar.python.types.InferredTypes.typeClassLocation;

@Rule(key = "S5756")
public class NonCallableCalledCheck extends PythonSubscriptionCheck {

  // List of non callable types with unresolved type hierarchies
  private static final List<String> NON_CALLABLE_TYPES = Arrays.asList("set", "frozenset");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Expression callee = callExpression.callee();
      InferredType calleeType = callee.type();
      if (!calleeType.canHaveMember("__call__") || NON_CALLABLE_TYPES.stream().anyMatch(calleeType::canOnlyBe)) {
        String name = nameFromExpression(callee);
        PreciseIssue preciseIssue;
        if (name != null) {
          preciseIssue = ctx.addIssue(callee, String.format("Fix this call; \"%s\"%s is not callable.", name, addTypeName(calleeType)));
        } else {
          preciseIssue = ctx.addIssue(callee, String.format("Fix this call; this expression%s is not callable.", addTypeName(calleeType)));
        }
        LocationInFile location = typeClassLocation(calleeType);
        if (location != null) {
          preciseIssue.secondary(location, null);
        }
      }
    });
  }

  private static String nameFromExpression(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      return ((Name) expression).name();
    }
    return null;
  }

  private static String addTypeName(InferredType type) {
    String typeName = InferredTypes.typeName(type);
    if (typeName != null) {
      return " has type " + typeName + " and it";
    }
    return "";
  }
}
