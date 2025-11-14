/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S7483")
public class AsyncFunctionWithTimeoutCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove this \"timeout\" parameter and use a timeout context manager instead.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, AsyncFunctionWithTimeoutCheck::checkFunctionDef);
  }

  private static void checkFunctionDef(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();

    // Check if the function is async using its token
    Token asyncKeyword = functionDef.asyncKeyword();
    if (asyncKeyword == null) {
      return;
    }

    FunctionType functionType = (FunctionType) functionDef.name().typeV2();
    if (mightBeOverridingMethod(functionType)) {
      return;
    }

    for (Parameter parameter : TreeUtils.nonTupleParameters(functionDef)) {
      Name parameterName = parameter.name();
      if (parameterName != null && "timeout".equals(parameterName.name())) {
        ctx.addIssue(parameter, MESSAGE).secondary(asyncKeyword, "This function is async.");
        return;
      }
    }
  }

  private static boolean mightBeOverridingMethod(FunctionType functionType) {
    return functionType.owner() instanceof ClassType classType && (classType.hasUnresolvedHierarchy() || classType.inheritedMember(functionType.name()).isPresent());
  }
}
