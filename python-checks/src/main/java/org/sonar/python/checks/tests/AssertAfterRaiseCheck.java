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
package org.sonar.python.checks.tests;

import java.util.List;
import java.util.Objects;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.WithStatement;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tests.UnittestUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5915")
public class AssertAfterRaiseCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE_MULTIPLE_STATEMENT = "Don’t perform an assertion here; An exception is expected to be raised before its execution.";
  private static final String MESSAGE_SINGLE_STATEMENT = "Refactor this test; if this assertion’s argument raises an exception, the assertion will never get executed.";
  private static final String MESSAGE_SECONDARY = "An exception is expected to be raised in this block.";

  private static final String ASSERTION_ERROR = "AssertionError";
  private static final String PYTEST_RAISE_CALL = "pytest.raises";
  private static final String PYTEST_ARG_EXCEPTION = "expected_exception";
  private static final String UNITTEST_ARG_EXCEPTION = "exception";
  public static final String QUICK_FIX_MESSAGE = "Change indentation level";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.WITH_STMT, ctx -> {
      WithStatement withStatement = (WithStatement) ctx.syntaxNode();
      if (!isWithStatementItemARaise(withStatement)) {
        return;
      }

      List<Statement> statements = withStatement.statements().statements();
      Statement statement = statements.get(statements.size()-1);
      if (isAnAssert(statement)) {
        var message = statements.size() > 1 ? MESSAGE_MULTIPLE_STATEMENT : MESSAGE_SINGLE_STATEMENT;
        var issue = ctx.addIssue(statement, message)
          .secondary(IssueLocation.preciseLocation(withStatement.firstToken(), withStatement.colon(), MESSAGE_SECONDARY));

        if (statements.size() > 1) {
          var quickFix = PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
            .addTextEdit(createTextEdits(withStatement, statement))
            .build();
          issue.addQuickFix(quickFix);
        }
      }
    });
  }

  private static List<PythonTextEdit> createTextEdits(WithStatement withStatement, Statement statement) {
    if (statement.firstToken().line() == withStatement.firstToken().line()) {
      var textToInsert = "\n" + " ".repeat(withStatement.firstToken().column());
      return List.of(TextEditUtils.insertBefore(statement, textToInsert));
    } else {
      int offset = statement.firstToken().column() - withStatement.firstToken().column();
      return TextEditUtils.shiftLeft(statement, offset);
    }
  }

  public boolean isWithStatementItemARaise(WithStatement withStatement) {
    return withStatement.withItems().stream()
      .filter(withItem -> withItem.test().is(Tree.Kind.CALL_EXPR))
      .map(withItem -> (CallExpression) withItem.test())
      .anyMatch(callExpression -> isValidPytestRaise(callExpression) || isValidUnittestRaise(callExpression));
  }

  public boolean isValidPytestRaise(CallExpression callExpression) {
    return Optional.of(callExpression).stream()
      .map(call -> TreeUtils.getSymbolFromTree(call.callee()))
      .filter(Optional::isPresent)
      .map(Optional::get)
      .map(Symbol::fullyQualifiedName)
      .filter(Objects::nonNull)
      .anyMatch(fqn -> fqn.contains(PYTEST_RAISE_CALL))
    && isNotAssertionErrorArgument(TreeUtils.nthArgumentOrKeyword(0, PYTEST_ARG_EXCEPTION, callExpression.arguments()));
  }

  public boolean isValidUnittestRaise(CallExpression callExpression) {
    return Optional.of(callExpression).stream()
      .filter(call -> call.callee().is(Tree.Kind.QUALIFIED_EXPR))
      .map(call -> (QualifiedExpression) call.callee())
      .anyMatch(
        callee -> callee.qualifier().is(Tree.Kind.NAME)
        && ((Name) callee.qualifier()).name().equals("self")
        && UnittestUtils.RAISE_METHODS.contains(callee.name().name()))
      && isNotAssertionErrorArgument(TreeUtils.nthArgumentOrKeyword(0, UNITTEST_ARG_EXCEPTION, callExpression.arguments()));
  }

  public boolean isNotAssertionErrorArgument(RegularArgument regularArgument) {
    return Optional.ofNullable(regularArgument).stream()
      .filter(Objects::nonNull)
      .map(arg -> TreeUtils.getSymbolFromTree(arg.expression()))
      .anyMatch(optSym -> optSym.isEmpty() || !ASSERTION_ERROR.equals(optSym.get().fullyQualifiedName()));
  }

  public boolean isAnAssert(Statement statement) {
    if (statement.is(Tree.Kind.ASSERT_STMT)) {
      return true;
    }

    return Optional.of(statement).stream()
      .filter(stat -> stat.is(Tree.Kind.EXPRESSION_STMT))
      .map(ExpressionStatement.class::cast)
      .map(ExpressionStatement::expressions)
      .anyMatch(expressions ->
        expressions.stream()
          .filter(expression -> expression.is(Tree.Kind.CALL_EXPR))
          .map(expression -> ((CallExpression) expression).callee())
          .filter(callee -> callee.is(Tree.Kind.QUALIFIED_EXPR))
          .map(QualifiedExpression.class::cast)
          .anyMatch(this::isUnittestAssert)
      );
  }

  public boolean isUnittestAssert(QualifiedExpression callee) {
    return callee.qualifier().is(Tree.Kind.NAME) && ((Name) callee.qualifier()).name().equals("self")
       && UnittestUtils.allAssertMethods().contains(callee.name().name());
  }
}
