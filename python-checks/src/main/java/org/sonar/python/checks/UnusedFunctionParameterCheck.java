/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.checks;

import java.util.List;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.semantic.SymbolUtils;
import org.sonar.python.tree.FunctionDefImpl;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S1172")
public class UnusedFunctionParameterCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove the unused function parameter \"%s\".";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FUNCDEF, ctx -> checkFunctionParameter(ctx, ((FunctionDef) ctx.syntaxNode())));
  }

  private static void checkFunctionParameter(SubscriptionContext ctx, FunctionDef functionDef) {
    if (isException(ctx, functionDef)) return;
    functionDef.localVariables().stream()
      .filter(symbol -> !isIgnoredSymbolName(symbol.name()))
      .map(Symbol::usages)
      .filter(usages -> usages.size() == 1 && usages.get(0).tree().parent().is(Kind.PARAMETER))
      .map(usages -> (Parameter) usages.get(0).tree().parent())
      .forEach(param -> ctx.addIssue(param, String.format(MESSAGE, param.name().name())));
  }

  private static boolean isIgnoredSymbolName(String symbolName) {
    return "self".equals(symbolName) || symbolName.startsWith("_");
  }

  private static boolean isException(SubscriptionContext ctx, FunctionDef functionDef) {
    FunctionSymbol functionSymbol = ((FunctionDefImpl) functionDef).functionSymbol();
    return CheckUtils.containsCallToLocalsFunction(functionDef) ||
      SymbolUtils.canBeAnOverridingMethod(functionSymbol) ||
      isInterfaceMethod(functionDef) ||
      isNotImplemented(functionDef) ||
      !functionDef.decorators().isEmpty() ||
      isSpecialMethod(functionDef) ||
      hasNonCallUsages(functionSymbol) ||
      isTestFunction(ctx, functionDef);
  }

  private static boolean isInterfaceMethod(FunctionDef functionDef) {
    return functionDef.body().statements().stream()
      .allMatch(statement -> statement.is(Kind.PASS_STMT, Kind.RAISE_STMT)
        || (statement.is(Kind.EXPRESSION_STMT) && isStringExpressionOrEllipsis((ExpressionStatement) statement)));
  }

  // Note that this will also exclude method containing only a return statement that returns nothing
  private static boolean isNotImplemented(FunctionDef functionDef) {
    List<Statement> statements = functionDef.body().statements();
    if (statements.size() != 1) return false;
    if (!statements.get(0).is(Kind.RETURN_STMT)) return false;
    ReturnStatement returnStatement = (ReturnStatement) statements.get(0);
    return returnStatement.expressions().stream().allMatch(retValue ->
      TreeUtils.getSymbolFromTree(retValue).filter(s -> "NotImplemented".equals(s.fullyQualifiedName())).isPresent());
  }

  private static boolean isStringExpressionOrEllipsis(ExpressionStatement stmt) {
    return stmt.expressions().stream().allMatch(expr -> expr.is(Kind.STRING_LITERAL, Kind.ELLIPSIS));
  }

  private static boolean isSpecialMethod(FunctionDef functionDef) {
    String name = functionDef.name().name();
    return name.startsWith("__") && name.endsWith("__");
  }

  private static boolean hasNonCallUsages(@Nullable FunctionSymbol functionSymbol) {
    return Optional.ofNullable(functionSymbol)
      .filter(fs -> fs.usages().stream().anyMatch(usage -> usage.kind() != Usage.Kind.FUNC_DECLARATION && !isFunctionCall(usage)))
      .isPresent();
  }

  private static boolean isTestFunction(SubscriptionContext ctx, FunctionDef functionDef) {
    String fileName = ctx.pythonFile().fileName();
    if (fileName.startsWith("conftest") || fileName.startsWith("test")) {
      return true;
    }
    return functionDef.name().name().startsWith("test");
  }

  private static boolean isFunctionCall(Usage usage) {
    if (usage.kind() != Usage.Kind.OTHER) return false;
    Tree tree = usage.tree();
    CallExpression callExpression = ((CallExpression) TreeUtils.firstAncestorOfKind(tree, Kind.CALL_EXPR));
    if (callExpression == null) return false;
    Expression callee = callExpression.callee();
    return callee == tree || TreeUtils.hasDescendant(callee, t -> t == tree);
  }
}
