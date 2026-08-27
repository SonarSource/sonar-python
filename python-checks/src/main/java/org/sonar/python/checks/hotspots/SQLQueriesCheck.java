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
package org.sonar.python.checks.hotspots;

import java.util.List;
import java.util.Optional;
import java.util.Set;
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
  private static final Set<String> ORACLEDB_SINK_METHODS = Set.of("execute", "executemany", "parse", "prepare", "fetch_df_all", "fetch_df_batches");
  private static final Set<String> ORACLEDB_MODULE_FQNS = Set.of("oracledb", "cx_Oracle");
  private static final Set<String> ORACLEDB_CONNECT_FQNS = Set.of("oracledb.connect", "cx_Oracle.connect");
  private boolean isUsingDjangoModel = false;
  private boolean isUsingDjangoDBConnection = false;
  private boolean isUsingOracleDB = false;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::visitFile);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCallExpression);
  }

  private void visitFile(SubscriptionContext ctx) {
    FileInput tree = (FileInput) ctx.syntaxNode();
    DatabaseApiUsageVisitor visitor = new DatabaseApiUsageVisitor();
    tree.accept(visitor);
    isUsingDjangoModel = visitor.usesDjangoModel;
    isUsingDjangoDBConnection = visitor.usesDjangoDbConnection;
    isUsingOracleDB = visitor.usesOracleDb;
  }

  private static class DatabaseApiUsageVisitor extends BaseTreeVisitor {

    private boolean usesDjangoModel = false;
    private boolean usesDjangoDbConnection = false;
    private boolean usesOracleDb = false;

    @Override
    public void visitAliasedName(AliasedName aliasedName) {
      Name boundName = aliasedName.alias();
      if (boundName == null) {
        boundName = aliasedName.dottedName().names().get(0);
      }
      Symbol symbol = boundName.symbol();
      String fullyQualifiedName = symbol != null ? symbol.fullyQualifiedName() : null;
      if (fullyQualifiedName != null) {
        if (fullyQualifiedName.contains("django.db.models")) {
          usesDjangoModel = true;
        }
        if (fullyQualifiedName.contains("django.db.connection")) {
          usesDjangoDbConnection = true;
        }
      }
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      if (isOracleDbConnectCall(callExpression)) {
        usesOracleDb = true;
      }
      super.visitCallExpression(callExpression);
    }

    /**
     * Recognizes both the {@code from oracledb import connect; connect(...)} idiom (the callee
     * symbol itself resolves to FQN "oracledb.connect") and the {@code import oracledb;
     * oracledb.connect(...)} idiom (no typeshed stub, so the qualifier only ever resolves to FQN
     * "oracledb", never "oracledb.connect" — matched by qualifier FQN plus attribute name instead).
     */
    private static boolean isOracleDbConnectCall(CallExpression callExpression) {
      Symbol calleeSymbol = callExpression.calleeSymbol();
      if (calleeSymbol != null && calleeSymbol.fullyQualifiedName() != null && ORACLEDB_CONNECT_FQNS.contains(calleeSymbol.fullyQualifiedName())) {
        return true;
      }
      if (callExpression.callee().is(Tree.Kind.QUALIFIED_EXPR)) {
        QualifiedExpression qualifiedExpression = (QualifiedExpression) callExpression.callee();
        if ("connect".equals(qualifiedExpression.name().name()) && qualifiedExpression.qualifier().is(Tree.Kind.NAME)) {
          Symbol qualifierSymbol = ((Name) qualifiedExpression.qualifier()).symbol();
          return qualifierSymbol != null && qualifierSymbol.fullyQualifiedName() != null && ORACLEDB_MODULE_FQNS.contains(qualifierSymbol.fullyQualifiedName());
        }
      }
      return false;
    }
  }

  private boolean isSQLQueryFromDjangoModel(String functionName) {
    return isUsingDjangoModel && ("raw".equals(functionName) || "extra".equals(functionName));
  }

  private boolean isSQLQueryFromDBConnection(String functionName) {
    if (isUsingDjangoDBConnection && "execute".equals(functionName)) {
      return true;
    }
    return isUsingOracleDB && ORACLEDB_SINK_METHODS.contains(functionName);
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
      if ((isSQLQueryFromDjangoModel(functionName) || isSQLQueryFromDBConnection(functionName))
        && !isException(callExpression, functionName)) {
        addIssue(context, callExpression);
      }
    }
  }

  private static void addIssue(SubscriptionContext context, CallExpression callExpression) {
    Optional<Tree> secondary = sensitiveArgumentValue(callExpression, context);
    secondary.ifPresent(tree -> context.addIssue(callExpression, MESSAGE).secondary(tree, null));
  }

  private static boolean isException(CallExpression callExpression, String functionName) {
    List<Argument> argListNode = callExpression.arguments();
    if (extraContainsFormattedSqlQueries(argListNode, functionName)) {
      return false;
    }
    return argListNode.isEmpty();
  }

  private static Optional<Tree> sensitiveArgumentValue(CallExpression callExpression, SubscriptionContext ctx) {
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
      return findFormattedValue((Name) expression, ctx);
    }
    if (isFormatted(expression)) {
      return Optional.of(expression);
    }
    return Optional.empty();
  }

  private static Optional<Tree> findFormattedValue(Name name, SubscriptionContext ctx) {
    Set<Expression> values = ctx.valuesAtLocation(name);
    if (!values.isEmpty()) {
      return values.stream()
        .filter(SQLQueriesCheck::isFormatted)
        .findFirst()
        .map(Tree.class::cast);
    }
    return Optional.ofNullable(Expressions.singleAssignedValue(name))
      .filter(SQLQueriesCheck::isFormatted)
      .map(Tree.class::cast);
  }

  private static boolean isFormatted(Expression tree) {
    FormattedStringVisitor visitor = new FormattedStringVisitor();
    tree.accept(visitor);
    return visitor.hasFormattedString;
  }

  private static boolean extraContainsFormattedSqlQueries(List<Argument> argListNode, String functionName) {
    if ("extra".equals(functionName)) {
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
        hasFormattedString |= "format".equals(callee.name().name()) && callee.qualifier().is(Tree.Kind.STRING_LITERAL);
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
