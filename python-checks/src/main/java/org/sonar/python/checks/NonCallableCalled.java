/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;

import static org.sonar.python.tree.TreeUtils.nameFromExpression;
import static org.sonar.python.types.InferredTypes.typeClassLocation;

public abstract class NonCallableCalled extends PythonSubscriptionCheck {
  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Expression callee = callExpression.callee();
      InferredType calleeType = callee.type();
      if (isNonCallableType(calleeType)) {
        String name = nameFromExpression(callee);
        PreciseIssue preciseIssue = ctx.addIssue(callee, message(calleeType, name));
        LocationInFile location = typeClassLocation(calleeType);
        if (location != null) {
          preciseIssue.secondary(location, "Definition.");
        }
      }
    });
  }

  protected static String addTypeName(InferredType type) {
    String typeName = InferredTypes.typeName(type);
    if (typeName != null) {
      return " has type " + typeName + " and it";
    }
    return "";
  }

  public abstract boolean isNonCallableType(InferredType type);

  public abstract String message(InferredType calleeType, @Nullable String name);

}
