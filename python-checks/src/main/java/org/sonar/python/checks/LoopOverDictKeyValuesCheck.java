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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.types.v2.TriBool;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7517")
public class LoopOverDictKeyValuesCheck extends PythonSubscriptionCheck {
  public static final String DICT_FQN = "dict";
  public static final String MESSAGE = "Use items to iterate over key-value pairs";
  private TypeCheckBuilder dictTypeCheck;


  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initChecks);
    context.registerSyntaxNodeConsumer(Tree.Kind.FOR_STMT, this::checkForStatement);
    context.registerSyntaxNodeConsumer(Tree.Kind.COMP_FOR, this::checkComprehensionFor);
  }

  private void initChecks(SubscriptionContext ctx) {
    dictTypeCheck = ctx.typeChecker().typeCheckBuilder().isInstanceOf(DICT_FQN);
  }

  private void checkForStatement(SubscriptionContext ctx) {
    var forStatement = (ForStatement) ctx.syntaxNode();
    var expressions = forStatement.expressions();
    var testExpressions = forStatement.testExpressions();
    if (expressions.size() == 2
        && testExpressions.size() == 1
        && dictTypeCheck.check(testExpressions.get(0).typeV2()) == TriBool.TRUE) {
      ctx.addIssue(testExpressions.get(0), MESSAGE);
    }
  }

  private void checkComprehensionFor(SubscriptionContext ctx) {
    var comprehensionFor = (ComprehensionFor) ctx.syntaxNode();
    if (comprehensionFor.loopExpression() instanceof Tuple tuple
        && tuple.elements().size() == 2
        && dictTypeCheck.check(comprehensionFor.iterable().typeV2()) == TriBool.TRUE) {
      ctx.addIssue(comprehensionFor.iterable(), MESSAGE);
    }
  }
}
