/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.Trivia;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.checks.utils.StringLiteralValuesCollector;
import org.sonar.python.semantic.SymbolUtils;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.UsageV2;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.TriBool;

@Rule(key = "S1172")
public class UnusedFunctionParameterCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove the unused function parameter \"%s\".";

  private static final Set<String> AWS_LAMBDA_PARAMETERS = Set.of("event", "context");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FUNCDEF, ctx -> checkFunctionParameter(ctx, ((FunctionDef) ctx.syntaxNode())));
  }

  private static void checkFunctionParameter(SubscriptionContext ctx, FunctionDef functionDef) {
    if (isException(ctx, functionDef)) return;
    functionDef.localVariables().stream()
      .filter(symbol -> !isIgnoredSymbolName(symbol.name()))
      .filter(UnusedFunctionParameterCheck::isUnused)
      .filter(symbol -> !isUsedInStringLiteralOrComment(symbol.name(), functionDef))
      .map(symbol -> (Parameter) symbol.usages().get(0).tree().parent())
      .forEach(param -> ctx.addIssue(param, String.format(MESSAGE, param.name().name())));
  }

  private static boolean isUnused(Symbol s) {
    return s.usages().size() == 1 && s.usages().get(0).tree().parent().is(Kind.PARAMETER);
  }

  /* If the parameter is used within a string literal or comment, this might indicate either:
   * A docstring or a comment explains why this parameter is unused (e.g the method is method to be overridden)
   * The parameter is used through a DSL (e.g pandas.DataFrame.query)
   */
  private static boolean isUsedInStringLiteralOrComment(String symbolName, FunctionDef functionDef) {
    StringLiteralValuesCollector stringLiteralValuesCollector = new StringLiteralValuesCollector();
    stringLiteralValuesCollector.collect(functionDef);
    List<String> comments = collectComments(functionDef);
    Pattern p = Pattern.compile("(^|\\s+|\"|'|@)" + symbolName + "($|\\s+|\"|')");
    return stringLiteralValuesCollector.anyMatches(str -> p.matcher(str).find()) ||
      comments.stream().anyMatch(str -> p.matcher(str).find());
  }

  private static boolean isIgnoredSymbolName(String symbolName) {
    return "self".equals(symbolName) || symbolName.startsWith("_") || AWS_LAMBDA_PARAMETERS.contains(symbolName);
  }

  private static boolean isException(SubscriptionContext ctx, FunctionDef functionDef) {
    return CheckUtils.containsCallToLocalsFunction(functionDef) ||
      SymbolUtils.canBeAnOverridingMethod(((FunctionType) functionDef.name().typeV2()), functionDef.name().firstToken().line()) ||
      isInterfaceMethod(functionDef) ||
      isNotImplemented(functionDef, ctx) ||
      !functionDef.decorators().isEmpty() ||
      isSpecialMethod(functionDef) ||
      hasNonCallUsages(functionDef) ||
      isTestFunction(ctx, functionDef) ||
      isAbstractClass(functionDef, ctx);
  }

  private static boolean isAbstractClass(FunctionDef functionDef, SubscriptionContext ctx) {
    var parentClassDef = CheckUtils.getParentClassDef(functionDef);
    if (parentClassDef == null) {
      return false;
    }
    var typeChecker = ctx.typeChecker().typeCheckBuilder().inheritsFrom("abc.ABC");
    return ((parentClassDef.name().typeV2() instanceof ClassType parentClassType) && parentClassType.hasMetaClass())
      || (typeChecker.check(parentClassDef.name().typeV2()) == TriBool.TRUE);
  }

  private static boolean isInterfaceMethod(FunctionDef functionDef) {
    return functionDef.body().statements().stream()
      .allMatch(statement -> statement.is(Kind.PASS_STMT, Kind.RAISE_STMT)
        || (statement.is(Kind.EXPRESSION_STMT) && isStringExpressionOrEllipsis((ExpressionStatement) statement)));
  }

  // Note that this will also exclude method containing only a return statement that returns nothing
  private static boolean isNotImplemented(FunctionDef functionDef, SubscriptionContext subscriptionContext) {
    List<Statement> statements = functionDef.body().statements();
    if (statements.size() != 1) return false;
    if (!statements.get(0).is(Kind.RETURN_STMT)) return false;
    ReturnStatement returnStatement = (ReturnStatement) statements.get(0);
    var typeChecker = subscriptionContext.typeChecker().typeCheckBuilder().isTypeWithName("NotImplemented");
    return returnStatement.expressions().stream().allMatch(expr -> typeChecker.check(expr.typeV2()) == TriBool.TRUE);
  }

  private static boolean isStringExpressionOrEllipsis(ExpressionStatement stmt) {
    return stmt.expressions().stream().allMatch(expr -> expr.is(Kind.STRING_LITERAL, Kind.ELLIPSIS));
  }

  private static boolean isSpecialMethod(FunctionDef functionDef) {
    String name = functionDef.name().name();
    return name.startsWith("__") && name.endsWith("__");
  }

  private static boolean hasNonCallUsages(FunctionDef functionDef) {
    return Optional.ofNullable(functionDef.name().symbolV2()).map(SymbolV2::usages)
      .filter(usages -> usages.stream().anyMatch(usage -> usage.kind() != UsageV2.Kind.FUNC_DECLARATION && !isFunctionCall(usage)))
      .isPresent();
  }

  private static boolean isFunctionCall(UsageV2 usage) {
    if (usage.kind() != UsageV2.Kind.OTHER) {
      return false;
    }
    var tree = usage.tree();
    var callExpression = ((CallExpression) TreeUtils.firstAncestorOfKind(tree, Kind.CALL_EXPR));
    if (callExpression == null) {
      return false;
    }
    var callee = callExpression.callee();
    return callee == tree || TreeUtils.hasDescendant(callee, t -> t == tree);

  }

  private static boolean isTestFunction(SubscriptionContext ctx, FunctionDef functionDef) {
    String fileName = ctx.pythonFile().fileName();
    if (fileName.startsWith("conftest") || fileName.startsWith("test")) {
      return true;
    }
    return functionDef.name().name().startsWith("test");
  }

  private static List<String> collectComments(Tree element) {
    List<String> comments = new ArrayList<>();
    Deque<Tree> stack = new ArrayDeque<>();
    stack.push(element);
    while (!stack.isEmpty()) {
      Tree currentElement = stack.pop();
      if (currentElement.is(Kind.TOKEN)) {
        ((Token) currentElement).trivia().stream().map(Trivia::value).forEach(comments::add);
      }
      for (int i = currentElement.children().size() - 1; i >= 0; i--) {
        Optional.ofNullable(currentElement.children().get(i)).ifPresent(stack::push);
      }
    }
    return comments;
  }
}
