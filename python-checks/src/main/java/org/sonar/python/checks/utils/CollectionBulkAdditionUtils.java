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
package org.sonar.python.checks.utils;

import java.util.ArrayList;
import java.util.List;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonCheck.PreciseIssue;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.python.tree.TreeUtils;

/**
 * Detection of collections that are grown one element at a time when a bulk method exists,
 * either through a for-loop or through a run of consecutive calls.
 * Shared by the rules covering {@code list.extend()} and {@code set.update()}.
 */
public final class CollectionBulkAdditionUtils {

  private static final int MIN_CONSECUTIVE_ADDITIONS = 3;

  private static final String FOR_LOOP_MESSAGE = "Use \"%s.%s()\" instead of a for-loop with \"%s()\".";
  private static final String CONSECUTIVE_MESSAGE = "Use \"%s.%s()\" instead of consecutive \"%s()\" calls.";
  private static final String SECONDARY_MESSAGE = "This call is part of the same sequence.";

  private CollectionBulkAdditionUtils() {
  }

  /**
   * Describes the collection a rule is about: the type of the receiver, the method adding a single
   * element, and the method adding them all at once.
   */
  public record BulkAddition(String typeName, TypeMatcher receiverType, TypeMatcher elementMethodMatcher, String elementMethod, String bulkMethod) {
  }

  private record Addition(QualifiedExpression call, Name receiver) {
  }

  public static void checkForStatement(SubscriptionContext ctx, BulkAddition bulkAddition) {
    ForStatement forStatement = (ForStatement) ctx.syntaxNode();

    // The bulk methods only accept synchronous iterables, so "async for" has no equivalent
    if (forStatement.isAsync()
      || forStatement.elseClause() != null
      || forStatement.expressions().size() != 1
      || forStatement.testExpressions().size() != 1) {
      return;
    }
    if (!(forStatement.expressions().get(0) instanceof Name loopVariable)) {
      return;
    }
    Expression iterable = forStatement.testExpressions().get(0);

    CallExpression call = singleBodyCall(forStatement);
    if (call == null || !(call.callee() instanceof QualifiedExpression callee)) {
      return;
    }
    if (!bulkAddition.elementMethodMatcher().isTrueFor(callee.name(), ctx)) {
      return;
    }
    Expression receiver = callee.qualifier();
    // Exclude transient objects like get_list().append(item)
    if (receiver instanceof CallExpression || !bulkAddition.receiverType().isTrueFor(receiver, ctx)) {
      return;
    }
    // Growing the very collection being iterated does not terminate, the bulk call is not equivalent
    if (isSameVariable(receiver, iterable)) {
      return;
    }
    Expression element = singlePositionalArgument(call);
    if (!(element instanceof Name argument) || !isSameVariable(loopVariable, argument)) {
      return;
    }

    ctx.addIssue(callee, message(FOR_LOOP_MESSAGE, bulkAddition));
  }

  public static void checkStatementList(SubscriptionContext ctx, BulkAddition bulkAddition) {
    StatementList statementList = (StatementList) ctx.syntaxNode();

    List<Addition> run = new ArrayList<>();
    for (Statement statement : statementList.statements()) {
      Addition addition = extractAddition(statement, bulkAddition, ctx);
      if (addition == null) {
        reportRun(ctx, bulkAddition, run);
        run.clear();
      } else {
        if (!run.isEmpty() && !isSameVariable(run.get(0).receiver(), addition.receiver())) {
          reportRun(ctx, bulkAddition, run);
          run.clear();
        }
        run.add(addition);
      }
    }
    reportRun(ctx, bulkAddition, run);
  }

  @CheckForNull
  private static Addition extractAddition(Statement statement, BulkAddition bulkAddition, SubscriptionContext ctx) {
    if (!(statement instanceof ExpressionStatement expressionStatement)
      || expressionStatement.expressions().size() != 1
      || !(expressionStatement.expressions().get(0) instanceof CallExpression call)
      || !(call.callee() instanceof QualifiedExpression callee)) {
      return null;
    }
    if (!bulkAddition.elementMethodMatcher().isTrueFor(callee.name(), ctx)) {
      return null;
    }
    // Only a plain variable can be tracked across statements, and it excludes transient receivers
    if (!(callee.qualifier() instanceof Name receiver) || !bulkAddition.receiverType().isTrueFor(receiver, ctx)) {
      return null;
    }
    Expression element = singlePositionalArgument(call);
    // An element reading the collection observes the previous additions, a bulk call would not
    if (element == null || readsVariable(element, receiver)) {
      return null;
    }
    return new Addition(callee, receiver);
  }

  private static void reportRun(SubscriptionContext ctx, BulkAddition bulkAddition, List<Addition> run) {
    if (run.size() < MIN_CONSECUTIVE_ADDITIONS) {
      return;
    }
    PreciseIssue issue = ctx.addIssue(run.get(0).call(), message(CONSECUTIVE_MESSAGE, bulkAddition));
    run.stream().skip(1).forEach(addition -> issue.secondary(addition.call(), SECONDARY_MESSAGE));
  }

  private static String message(String template, BulkAddition bulkAddition) {
    return String.format(template, bulkAddition.typeName(), bulkAddition.bulkMethod(), bulkAddition.elementMethod());
  }

  @CheckForNull
  private static CallExpression singleBodyCall(ForStatement forStatement) {
    List<Statement> statements = forStatement.body().statements();
    if (statements.size() != 1
      || !(statements.get(0) instanceof ExpressionStatement expressionStatement)
      || expressionStatement.expressions().size() != 1
      || !(expressionStatement.expressions().get(0) instanceof CallExpression call)) {
      return null;
    }
    return call;
  }

  @CheckForNull
  private static Expression singlePositionalArgument(CallExpression call) {
    if (call.arguments().size() != 1
      || !(call.arguments().get(0) instanceof RegularArgument argument)
      || argument.keywordArgument() != null
      || argument.expression() instanceof UnpackingExpression) {
      return null;
    }
    return argument.expression();
  }

  private static boolean readsVariable(Expression expression, Name variable) {
    return TreeUtils.hasDescendant(expression, tree -> isSameVariable(variable, tree))
      || isSameVariable(variable, expression);
  }

  private static boolean isSameVariable(@Nullable Tree first, @Nullable Tree second) {
    if (!(first instanceof Name firstName) || !(second instanceof Name secondName)) {
      return false;
    }
    SymbolV2 symbol = firstName.symbolV2();
    return symbol != null && symbol.equals(secondName.symbolV2());
  }
}
