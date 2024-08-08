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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.TypeChecker;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;

import static org.sonar.python.tree.TreeUtils.nameFromExpression;

@Rule(key = "S5756")
public class NonCallableCalledCheck extends PythonSubscriptionCheck {
  private static final Logger LOG = LoggerFactory.getLogger(NonCallableCalledCheck.class);

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Expression callee = callExpression.callee();
      PythonType type = callee.typeV2();
      if (isNonCallableType(type, ctx.typeChecker())) {
        if (callee instanceof CallExpression calleeCallExpression
            && calleeCallExpression.callee().typeV2() instanceof FunctionType functionType) {
          LOG.info("Callee is a function with type: " + functionType.name() + ", and type origin: " + functionType.typeOrigin().toString());
        }
        String name = nameFromExpression(callee);
        PreciseIssue preciseIssue = ctx.addIssue(callee, message(type, name));
        type.definitionLocation()
          .ifPresent(location -> preciseIssue.secondary(location, "Definition."));
      }
    });
  }

  protected static String addTypeName(PythonType type) {
    return type.displayName()
      .map(d -> " has type " + d + " and it")
      .orElse("");
  }

  public boolean isNonCallableType(PythonType type, TypeChecker typeChecker) {
    return typeChecker.typeCheckBuilder().hasMember("__call__").check(type) == TriBool.FALSE;
  }

  public String message(PythonType typeV2, @Nullable String name) {
    if (name != null) {
      return "Fix this call; \"%s\"%s is not callable.".formatted(name, addTypeName(typeV2));
    }
    return "Fix this call; this expression%s is not callable.".formatted(addTypeName(typeV2));
  }
}
