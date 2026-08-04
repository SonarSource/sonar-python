/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks.tests;

import java.util.List;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S9117")
public class PytestDefaultFixtureScopeCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove this redundant scope=\"function\" argument.";
  private static final String QUICK_FIX_MESSAGE = "Remove scope=\"function\"";
  private static final String SCOPE_ARGUMENT = "scope";
  private static final String FUNCTION_SCOPE = "function";

  private static final TypeMatcher PYTEST_FIXTURE_MATCHER = TypeMatchers.isType("pytest.fixture");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.DECORATOR, PytestDefaultFixtureScopeCheck::checkDecorator);
  }

  private static void checkDecorator(SubscriptionContext ctx) {
    Decorator decorator = (Decorator) ctx.syntaxNode();
    if (!(decorator.expression() instanceof CallExpression callExpression)) {
      return;
    }
    if (!PYTEST_FIXTURE_MATCHER.isTrueFor(callExpression.callee(), ctx)) {
      return;
    }

    RegularArgument scopeArgument = TreeUtils.argumentByKeyword(SCOPE_ARGUMENT, callExpression.arguments());
    if (scopeArgument == null || !isFunctionScope(scopeArgument.expression())) {
      return;
    }

    var issue = ctx.addIssue(scopeArgument, MESSAGE);
    createQuickFix(callExpression, scopeArgument).ifPresent(issue::addQuickFix);
  }

  private static boolean isFunctionScope(Expression expression) {
    return expression instanceof StringLiteral stringLiteral
      && FUNCTION_SCOPE.equals(stringLiteral.trimmedQuotesValue());
  }

  private static Optional<PythonQuickFix> createQuickFix(CallExpression callExpression, RegularArgument scopeArgument) {
    List<Argument> arguments = callExpression.arguments();
    int argIndex = arguments.indexOf(scopeArgument);
    if (argIndex == -1 || callExpression.argumentList() == null) {
      return Optional.empty();
    }

    if (arguments.size() == 1) {
      Token calleeEnd = callExpression.callee().lastToken();
      Token rightPar = callExpression.rightPar();
      return Optional.of(PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
        .addTextEdit(TextEditUtils.removeRange(
          calleeEnd.pythonLine(),
          endColumn(calleeEnd),
          rightPar.pythonLine(),
          endColumn(rightPar)))
        .build());
    }

    List<Tree> children = callExpression.argumentList().children();
    int childIndex = children.indexOf(scopeArgument);
    if (childIndex < 0) {
      return Optional.empty();
    }

    if (argIndex == 0) {
      // First argument: remove through the following comma. Prefer token-based removal when the
      // next argument is on the same line, or when scope shares its line with '(' (so we do not
      // delete the decorator). When scope is alone on its line, drop that whole line to keep the
      // next argument's indentation.
      Argument nextArg = arguments.get(1);
      Token scopeToken = scopeArgument.firstToken();
      Token nextToken = nextArg.firstToken();
      if (scopeToken.line() == nextToken.line() || callExpression.leftPar().line() == scopeToken.line()) {
        return Optional.of(PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
          .addTextEdit(TextEditUtils.removeUntil(scopeArgument, nextArg))
          .build());
      }
      return Optional.of(PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
        .addTextEdit(TextEditUtils.removeRange(
          scopeToken.pythonLine(),
          0,
          nextToken.pythonLine(),
          0))
        .build());
    }

    // Not first: remove the preceding comma and this argument (keep any following comma / next-arg indent)
    Tree removeFrom = children.get(childIndex - 1);
    if (argIndex == arguments.size() - 1) {
      return Optional.of(PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
        .addTextEdit(TextEditUtils.removeUntil(removeFrom, callExpression.rightPar()))
        .build());
    }

    Tree removeTo = children.get(childIndex + 1);
    return Optional.of(PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
      .addTextEdit(TextEditUtils.removeUntil(removeFrom, removeTo))
      .build());
  }

  private static int endColumn(Token token) {
    return token.pythonColumn() + token.value().length();
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }
}
