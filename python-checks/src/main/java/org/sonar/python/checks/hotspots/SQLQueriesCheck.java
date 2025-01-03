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
package org.sonar.python.checks.hotspots;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentExpression;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.plugins.python.api.symbols.Symbol;

@Rule(key = SQLQueriesCheck.CHECK_KEY)
public class SQLQueriesCheck extends PythonSubscriptionCheck {
  public static final String CHECK_KEY = "S2077";
  private static final String MESSAGE = "Make sure that formatting this SQL query is safe here.";
  private boolean isUsingDjangoModel = false;
  private boolean isUsingDjangoDBConnection = false;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::visitFile);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCallExpression);
  }

  private void visitFile(SubscriptionContext ctx) {
    isUsingDjangoModel = false;
    isUsingDjangoDBConnection = false;
    FileInput tree = (FileInput) ctx.syntaxNode();
    SymbolsFromImport visitor = new SymbolsFromImport();
    tree.accept(visitor);
    visitor.symbols.stream()
      .filter(Objects::nonNull)
      .map(Symbol::fullyQualifiedName)
      .filter(Objects::nonNull)
      .forEach(qualifiedName -> {
        if (qualifiedName.contains("django.db.models")) {
          isUsingDjangoModel = true;
        }
        if (qualifiedName.contains("django.db.connection")) {
          isUsingDjangoDBConnection = true;
        }
      });
  }

  private static class SymbolsFromImport extends BaseTreeVisitor {

    private Set<Symbol> symbols = new HashSet<>();

    @Override
    public void visitAliasedName(AliasedName aliasedName) {
      List<Name> names = aliasedName.dottedName().names();
      symbols.add(names.get(names.size() - 1).symbol());
    }
  }

  private boolean isSQLQueryFromDjangoModel(String functionName) {
    return isUsingDjangoModel && (functionName.equals("raw") || functionName.equals("extra"));
  }

  private boolean isSQLQueryFromDjangoDBConnection(String functionName) {
    return isUsingDjangoDBConnection && functionName.equals("execute");
  }

  private void checkCallExpression(SubscriptionContext context) {
    CallExpression callExpression = (CallExpression) context.syntaxNode();

    Symbol symbol = callExpression.calleeSymbol();
    if (symbol != null && "django.db.models.expressions.RawSQL".equals(symbol.fullyQualifiedName())) {
      addIssue(context, callExpression);
      return;
    }

    if (callExpression.callee().is(Tree.Kind.QUALIFIED_EXPR)) {
      String functionName = ((QualifiedExpression) callExpression.callee()).name().name();
      if ((isSQLQueryFromDjangoModel(functionName) || isSQLQueryFromDjangoDBConnection(functionName))
        && !isException(callExpression, functionName)) {
        addIssue(context, callExpression);
      }
    }
  }

  private static void addIssue(SubscriptionContext context, CallExpression callExpression) {
    Optional<Tree> secondary = sensitiveArgumentValue(callExpression);
    secondary.ifPresent(tree ->  context.addIssue(callExpression, MESSAGE).secondary(tree, null));
  }

  private static boolean isException(CallExpression callExpression, String functionName) {
    List<Argument> argListNode = callExpression.arguments();
    if (extraContainsFormattedSqlQueries(argListNode, functionName)) {
      return false;
    }
    return argListNode.isEmpty();
  }

  private static Optional<Tree> sensitiveArgumentValue(CallExpression callExpression) {
    List<Argument> argListNode = callExpression.arguments();
    if (argListNode.isEmpty()) {
      return Optional.empty();
    }
    Argument arg = argListNode.get(0);
    if (!arg.is(Tree.Kind.REGULAR_ARGUMENT)) {
      return Optional.empty();
    }
    Expression expression = getExpression(((RegularArgument) arg).expression());
    if (expression.is(Tree.Kind.NAME)) {
      expression = Expressions.singleAssignedValue((Name) expression);
    }
    if (expression != null && isFormatted(expression)) {
      return Optional.of(expression);
    }
    return Optional.empty();
  }

  private static boolean isFormatted(Expression tree) {
    FormattedStringVisitor visitor = new FormattedStringVisitor();
    tree.accept(visitor);
    return visitor.hasFormattedString;
  }

  private static boolean extraContainsFormattedSqlQueries(List<Argument> argListNode, String functionName) {
    if (functionName.equals("extra")) {
      return argListNode.stream()
        .filter(arg -> arg.is(Tree.Kind.REGULAR_ARGUMENT))
        .map(RegularArgument.class::cast)
        .filter(SQLQueriesCheck::isAssignment)
        .map(RegularArgument::expression)
        .anyMatch(SQLQueriesCheck::isFormatted);
    }
    return false;
  }

  private static boolean isAssignment(RegularArgument arg) {
    return arg.equalToken() != null;
  }

  private static Expression getExpression(Expression expr) {
    expr = Expressions.removeParentheses(expr);
    if (expr.is(Tree.Kind.ASSIGNMENT_EXPRESSION)) {
      return getExpression(((AssignmentExpression) expr).expression());
    }
    return expr;
  }

  private static class FormattedStringVisitor extends BaseTreeVisitor {
    boolean hasFormattedString = false;

    @Override
    public void visitStringElement(StringElement stringElement) {
      super.visitStringElement(stringElement);
      hasFormattedString |= stringElement.isInterpolated();
    }

    @Override
    public void visitCallExpression(CallExpression pyCallExpressionTree) {
      if (pyCallExpressionTree.callee().is(Tree.Kind.QUALIFIED_EXPR)) {
        QualifiedExpression callee = (QualifiedExpression) pyCallExpressionTree.callee();
        hasFormattedString |= callee.name().name().equals("format") && callee.qualifier().is(Tree.Kind.STRING_LITERAL);
      }
      super.visitCallExpression(pyCallExpressionTree);
    }

    @Override
    public void visitBinaryExpression(BinaryExpression pyBinaryExpressionTree) {
      hasFormattedString |= pyBinaryExpressionTree.leftOperand().is(Tree.Kind.STRING_LITERAL) || pyBinaryExpressionTree.rightOperand().is(Tree.Kind.STRING_LITERAL);
      super.visitBinaryExpression(pyBinaryExpressionTree);
    }
  }
}
