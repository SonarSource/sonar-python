/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.checks.hotspots;

import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.AbstractCallExpressionCheck;
import org.sonar.python.semantic.Symbol;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;

@Rule(key = SQLQueriesCheck.CHECK_KEY)
public class SQLQueriesCheck extends AbstractCallExpressionCheck {
  public static final String CHECK_KEY = "S2077";
  private static final String MESSAGE = "Make sure that formatting this SQL query is safe here.";
  private boolean isUsingDjangoModel = false;
  private boolean isUsingDjangoDBConnection = false;

  @Override
  protected Set<String> functionsToCheck() {
    return Collections.singleton("django.db.models.expressions.RawSQL");
  }

  @Override
  protected String message() {
    return MESSAGE;
  }

  @Override
  public void initialize(Context context) {
    super.initialize(context);
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::visitFile);
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

  @Override
  public void visitNode(SubscriptionContext context) {
    CallExpression callExpressionTree = (CallExpression) context.syntaxNode();
    if(callExpressionTree.callee().is(Tree.Kind.QUALIFIED_EXPR)) {
      String functionName = ((QualifiedExpression) callExpressionTree.callee()).name().name();
      if ((isSQLQueryFromDjangoModel(functionName) || isSQLQueryFromDjangoDBConnection(functionName)) && !isException(callExpressionTree, functionName)) {
        context.addIssue(callExpressionTree, MESSAGE);
      }
    }
    super.visitNode(context);
  }

  private static boolean isException(CallExpression callExpression, String functionName) {
    List<Argument> argListNode = callExpression.arguments();
    if (extraContainsFormattedSqlQueries(argListNode, functionName)) {
      return false;
    }
    if (argListNode.isEmpty()) {
      return true;
    }
    Argument arg = argListNode.get(0);
    return !isFormatted(arg.expression());
  }

  @Override
  protected boolean isException(CallExpression callExpression) {
    return isException(callExpression, "");
  }

  private static boolean isFormatted(Expression tree) {
    FormattedStringVisitor visitor = new FormattedStringVisitor();
    tree.accept(visitor);
    return visitor.hasFormattedString;
  }

  private static boolean extraContainsFormattedSqlQueries(List<Argument> argListNode, String functionName) {
    if (functionName.equals("extra")) {
      return argListNode.stream()
        .filter(SQLQueriesCheck::isAssignment)
        .map(Argument::expression)
        .anyMatch(SQLQueriesCheck::isFormatted);
    }
    return false;
  }

  private static boolean isAssignment(Argument arg) {
    return arg.equalToken() != null;
  }

  private static class FormattedStringVisitor extends BaseTreeVisitor {
    boolean hasFormattedString = false;

    @Override
    public void visitStringLiteral(StringLiteral pyStringLiteralTree) {
      super.visitStringLiteral(pyStringLiteralTree);
      hasFormattedString |= pyStringLiteralTree.stringElements().stream().anyMatch(se -> se.prefix().equalsIgnoreCase("f"));
    }

    @Override
    public void visitCallExpression(CallExpression pyCallExpressionTree) {
      if(pyCallExpressionTree.callee().is(Tree.Kind.QUALIFIED_EXPR)) {
        QualifiedExpression callee = (QualifiedExpression) pyCallExpressionTree.callee();
        hasFormattedString |=  callee.name().name().equals("format") && callee.qualifier().is(Tree.Kind.STRING_LITERAL);
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
