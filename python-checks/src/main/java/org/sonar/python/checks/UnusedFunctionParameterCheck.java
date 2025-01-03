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

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;
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
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.Trivia;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.checks.utils.StringLiteralValuesCollector;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.semantic.SymbolUtils;
import org.sonar.python.tree.FunctionDefImpl;
import org.sonar.python.tree.TreeUtils;

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
    FunctionSymbol functionSymbol = ((FunctionDefImpl) functionDef).functionSymbol();
    return CheckUtils.containsCallToLocalsFunction(functionDef) ||
      SymbolUtils.canBeAnOverridingMethod(functionSymbol) ||
      isInterfaceMethod(functionDef) ||
      isNotImplemented(functionDef) ||
      !functionDef.decorators().isEmpty() ||
      isSpecialMethod(functionDef) ||
      hasNonCallUsages(functionSymbol) ||
      isTestFunction(ctx, functionDef) ||
      isAbstractClass(functionDef);
  }

  private static boolean isAbstractClass(FunctionDef functionDef) {
    FunctionSymbol functionSymbol = ((FunctionDefImpl) functionDef).functionSymbol();
    if (functionSymbol == null) {
      return false;
    }
    Symbol owner = ((FunctionSymbolImpl) functionSymbol).owner();
    return owner != null && ((((ClassSymbolImpl) owner).superClasses().stream().anyMatch(symbol -> "abc.ABC".equals(symbol.fullyQualifiedName())))
      || (((ClassSymbolImpl) owner).hasMetaClass()));
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
