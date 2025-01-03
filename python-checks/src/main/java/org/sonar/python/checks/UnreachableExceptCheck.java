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

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TryStatement;
import org.sonar.plugins.python.api.tree.Tuple;

@Rule(key = "S1045")
public class UnreachableExceptCheck extends PythonSubscriptionCheck {

  private static final String SECONDARY_MESSAGE = "Exceptions will be caught here.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.TRY_STMT, ctx -> {

      TryStatement tryStatement = (TryStatement) ctx.syntaxNode();
      Map<String, Expression> caughtTypes = new HashMap<>();

      for (ExceptClause exceptClause : tryStatement.exceptClauses()) {
        handleExceptClause(ctx, caughtTypes, exceptClause);
      }
    });
  }

  private static void handleExceptClause(SubscriptionContext ctx, Map<String, Expression> caughtTypes, ExceptClause exceptClause) {
    Map<String, Expression> caughtInExceptClause = new HashMap<>();
    Expression exceptionExpression = exceptClause.exception();
    if (exceptionExpression == null) {
      Expression baseExceptionExpression = caughtTypes.get("BaseException");
      if (baseExceptionExpression != null) {
        ctx.addIssue(exceptClause.exceptKeyword(), "Merge this bare \"except:\" with the \"BaseException\" one.")
          .secondary(baseExceptionExpression, SECONDARY_MESSAGE);
      }
      return;
    }
    if (exceptionExpression.is(Tree.Kind.TUPLE)) {
      Tuple tuple = (Tuple) exceptionExpression;
      for (Expression expression : tuple.elements()) {
        handleExceptionExpression(ctx, caughtTypes, expression, caughtInExceptClause);
      }
    } else {
      handleExceptionExpression(ctx, caughtTypes, exceptionExpression, caughtInExceptClause);
    }
    caughtInExceptClause.forEach(caughtTypes::putIfAbsent);
  }

  private static void handleExceptionExpression(SubscriptionContext ctx, Map<String, Expression> caughtTypes,
                                         Expression exceptionExpression, Map<String, Expression> caughtInExceptClause) {
    if (!(exceptionExpression instanceof HasSymbol hasSymbol)) {
      return;
    }
    var symbol = hasSymbol.symbol();
    if (symbol == null) {
      return;
    }
    var symbolName = getSymbolName(symbol).orElse(null);
    var handledExceptions = Symbol.Kind.CLASS == symbol.kind() ?
      retrieveAlreadyHandledExceptionsByClass((ClassSymbol) symbol, caughtTypes)
      : retrieveAlreadyHandledExceptionsByFullyQualifiedName(symbol, caughtTypes);

    if (!handledExceptions.isEmpty()) {
      var issue = ctx.addIssue(exceptionExpression, "Catch this exception only once; it is already handled by a previous except clause.");
      handledExceptions.forEach(h -> issue.secondary(h, SECONDARY_MESSAGE));
    }
    caughtInExceptClause.put(symbolName, exceptionExpression);
  }

  private static List<Expression> retrieveAlreadyHandledExceptionsByClass(ClassSymbol classSymbol, Map<String, Expression> caughtTypes) {
    return caughtTypes.keySet().stream().filter(classSymbol::isOrExtends).map(caughtTypes::get).toList();
  }

  private static List<Expression> retrieveAlreadyHandledExceptionsByFullyQualifiedName(Symbol symbol, Map<String, Expression> caughtTypes) {
    return getSymbolName(symbol)
      .filter(caughtTypes::containsKey)
      .map(caughtTypes::get)
      .stream()
      .toList();
  }

  private static Optional<String> getSymbolName(Symbol symbol) {
    return Optional.ofNullable(symbol.fullyQualifiedName())
      .or(() -> Optional.ofNullable(symbol.name()));
  }
}
