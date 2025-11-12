/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7507")
public class DefaultFactoryArgumentCheck extends PythonSubscriptionCheck {

  private TypeCheckBuilder defaultDictTypeChecker = null;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx ->
      defaultDictTypeChecker = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn("collections.defaultdict")
    );
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      if (defaultDictTypeChecker.check(callExpression.callee().typeV2()) != TriBool.TRUE) {
        return;
      }
      RegularArgument defaultFactoryKeywordArgument = TreeUtils.argumentByKeyword("default_factory", callExpression.arguments());
      if (defaultFactoryKeywordArgument != null) {
        ctx.addIssue(defaultFactoryKeywordArgument, "Replace this keyword argument with a positional argument at the first place.");
      }
    });
  }
}
