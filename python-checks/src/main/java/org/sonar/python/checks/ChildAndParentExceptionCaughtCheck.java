/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5713")
public class ChildAndParentExceptionCaughtCheck extends PythonSubscriptionCheck {
  public static final String QUICK_FIX_MESSAGE = "Remove the redundant Exception";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.EXCEPT_CLAUSE, ChildAndParentExceptionCaughtCheck::checkExceptClause);
    context.registerSyntaxNodeConsumer(Tree.Kind.EXCEPT_GROUP_CLAUSE, ChildAndParentExceptionCaughtCheck::checkExceptClause);
  }

  private static void checkExceptClause(SubscriptionContext ctx) {
    ExceptClause exceptClause = (ExceptClause) ctx.syntaxNode();
    Map<ClassSymbol, List<Expression>> caughtExceptionsBySymbol = new HashMap<>();
    Expression exceptionExpression = exceptClause.exception();
    if (exceptionExpression == null) {
      return;
    }
    TreeUtils.flattenTuples(exceptionExpression).forEach(e -> addExceptionExpression(e, caughtExceptionsBySymbol));
    checkCaughtExceptions(ctx, caughtExceptionsBySymbol);
  }

  private static void checkCaughtExceptions(SubscriptionContext ctx, Map<ClassSymbol, List<Expression>> caughtExceptionsBySymbol) {
    caughtExceptionsBySymbol.forEach((currentSymbol, caughtExceptionsWithSameSymbol) -> {
      Expression currentException = caughtExceptionsWithSameSymbol.get(0);
      if (caughtExceptionsWithSameSymbol.size() > 1) {
        var issue = ctx.addIssue(currentException, "Remove this duplicate Exception class.");
        addQuickFix(issue, currentException);
        caughtExceptionsWithSameSymbol.stream().skip(1).forEach(e -> issue.secondary(e, "Duplicate."));
      }

      var caughtParentExceptions = caughtExceptionsBySymbol.entrySet()
        .stream()
        .filter(entry -> entry.getKey() != currentSymbol && currentSymbol.isOrExtends(entry.getKey()))
        .toList();

      if (!caughtParentExceptions.isEmpty()) {
        var issue = ctx.addIssue(currentException, "Remove this redundant Exception class; it derives from another which is already caught.");
        addQuickFix(issue, currentException);

        caughtParentExceptions.stream()
          .map(Map.Entry::getValue)
          .forEach(entries -> addSecondaryLocations(issue, entries));
      }
    });
  }

  private static void addQuickFix(PreciseIssue issue, Expression currentException) {
    var quickFix = createQuickFix(currentException);
    if (quickFix != null) {
      issue.addQuickFix(quickFix);
    }
  }

  private static List<String> collectNamesFromTuple(Expression expression) {
    expression = Expressions.removeParentheses(expression);
    if (expression.is(Tree.Kind.TUPLE)) {
      var tuple = (Tuple) expression;
      return tuple.elements()
        .stream()
        .map(ChildAndParentExceptionCaughtCheck::collectNames)
        .flatMap(Collection::stream)
        .collect(Collectors.toList());
    }
    throw new IllegalArgumentException("Unsupported kind of tree element: " + expression.getKind().name());
  }
  private static List<String> collectNames(Expression expression) {
    expression = Expressions.removeParentheses(expression);
    if (expression.is(Tree.Kind.NAME)) {
      var name = (Name) expression;
      return List.of(name.name());
    } else if (expression.is(Tree.Kind.QUALIFIED_EXPR)) {
      var name = TreeUtils.tokens(expression)
        .stream()
        .map(Token::value)
        .collect(Collectors.joining());
      return List.of(name);
    }
    throw new IllegalArgumentException("Unsupported kind of tree element: " + expression.getKind().name());
  }

  private static PythonQuickFix createQuickFix(Expression currentException) {
    try {
      var currentExceptionName = collectNames(currentException).get(0);

      return Optional.of(currentException)
        .map(exception -> TreeUtils.firstAncestorOfKind(exception, Tree.Kind.EXCEPT_CLAUSE))
        .map(ExceptClause.class::cast)
        .map(ExceptClause::exception)
        .map(exceptions -> {
          List<String> names = collectNamesFromTuple(exceptions);
          names.remove(currentExceptionName);

          var text = names.size() == 1 ? names.get(0) : names.stream().collect(Collectors.joining(", ", "(", ")"));

          return PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
            .addTextEdit(TextEditUtils.replace(exceptions, text))
            .build();
        }).orElse(null);
    } catch (IllegalArgumentException e) {
      // expression contains subexpressions that are out of scope for quick fixing
      return null;
    }
  }

  private static void addExceptionExpression(Expression exceptionExpression, Map<ClassSymbol, List<Expression>> caughtExceptionsByFQN) {
    if (exceptionExpression instanceof HasSymbol hasSymbol) {
      Symbol symbol = hasSymbol.symbol();
      if (symbol != null && symbol.kind().equals(Symbol.Kind.CLASS)) {
        ClassSymbol classSymbol = (ClassSymbol) symbol;
        caughtExceptionsByFQN.computeIfAbsent(classSymbol, k -> new ArrayList<>()).add(exceptionExpression);
      }
    }
  }

  private static void addSecondaryLocations(PreciseIssue issue, List<Expression> others) {
    for (Expression other : others) {
      issue.secondary(other, "Parent class.");
    }
  }
}
