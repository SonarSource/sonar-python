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

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7517")
public class LoopOverDictKeyValuesCheck extends PythonSubscriptionCheck {
  private static final String DICT_FQN = "dict";
  private static final String MESSAGE = "Use items to iterate over key-value pairs";
  private static final String QUICK_FIX_MESSAGE = "Replace with items method call";
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
      var dict = testExpressions.get(0);
      var issue = ctx.addIssue(dict, MESSAGE);
      createQuickFix(dict).ifPresent(issue::addQuickFix);
    }
  }

  private void checkComprehensionFor(SubscriptionContext ctx) {
    var comprehensionFor = (ComprehensionFor) ctx.syntaxNode();
    if (comprehensionFor.loopExpression() instanceof Tuple tuple
        && tuple.elements().size() == 2
        && dictTypeCheck.check(comprehensionFor.iterable().typeV2()) == TriBool.TRUE) {
      var dict = comprehensionFor.iterable();
      var issue = ctx.addIssue(dict, MESSAGE);
      createQuickFix(dict).ifPresent(issue::addQuickFix);
    }
  }

  private static Optional<PythonQuickFix> createQuickFix(Expression dict) {
    return Optional.ofNullable(TreeUtils.treeToString(dict, false))
      .map("%s.items()"::formatted)
      .map(replacementText -> TextEditUtils.replace(dict, replacementText))
      .map(textEdit -> PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE).addTextEdit(textEdit))
      .map(PythonQuickFix.Builder::build);
  }
}
