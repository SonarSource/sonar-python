/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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

import java.util.regex.Pattern;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.AwsLambdaChecksUtils;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7620")
public class AwsLambdaTmpCleanupCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Clean up this temporary file before the Lambda function completes.";
  private static final String SECONDARY_LOCATION_MESSAGE = "The temporary folder is used in this Lambda function.";
  private static final Pattern TMP_PATH_PATTERN = Pattern.compile("^/tmp/.*");

  private TypeCheckBuilder openType;
  private TypeCheckBuilder osRemoveType;
  private TypeCheckBuilder osUnlinkType;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initializeTypeChecker);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::visitCallExpression);
  }

  private void initializeTypeChecker(SubscriptionContext ctx) {
    openType = ctx.typeChecker().typeCheckBuilder().isBuiltinWithName("open");
    osRemoveType = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn("os.remove");
    osUnlinkType = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn("os.unlink");
  }

  private void visitCallExpression(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();

    Tree functionDefAncestor = TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.FUNCDEF);
    if (functionDefAncestor == null) {
      return;
    }

    FunctionDef lambdaHandler = (FunctionDef) functionDefAncestor;

    if (!AwsLambdaChecksUtils.isLambdaHandler(ctx, lambdaHandler)) {
      return;
    }

    if (openType.check(callExpression.callee().typeV2()).isTrue() && !isTempFileCleanedUp(callExpression, lambdaHandler)) {
      ctx.addIssue(callExpression.callee(), MESSAGE)
        .secondary(lambdaHandler.name(), SECONDARY_LOCATION_MESSAGE);
    }
  }

  private boolean isTempFileCleanedUp(CallExpression callExpression, FunctionDef lambdaHandler) {
    RegularArgument regularArg = TreeUtils.nthArgumentOrKeyword(0, "file", callExpression.arguments());

    if (regularArg == null) {
      return true;
    }

    Expression pathExpression = regularArg.expression();
    String pathValue = getStringValue(pathExpression);

    return pathValue == null ||
      !TMP_PATH_PATTERN.matcher(pathValue).matches() ||
      hasCleanupCall(lambdaHandler, pathValue) ||
      isPathPassedToOtherFunction(pathExpression);
  }

  private boolean isPathPassedToOtherFunction(Expression pathExpression) {
    if (!(pathExpression instanceof Name)) {
      return true;
    }
    Name name = (Name) pathExpression;
    SymbolV2 symbol = name.symbolV2();
    if (symbol == null) {
      return true;
    }
    return getUsagesInCallExpr(symbol)
      .anyMatch(callExpr -> !isOsRemoveOrUnlinkCall(callExpr) && !openType.check(callExpr.callee().typeV2()).isTrue());

  }

  private static Stream<CallExpression> getUsagesInCallExpr(SymbolV2 symbol) {
    return symbol.usages().stream()
      .filter(usage -> usage.kind().equals(UsageV2.Kind.OTHER))
      .map(usage -> usage.tree().parent())
      .filter(parent -> parent.is(Tree.Kind.REGULAR_ARGUMENT))
      .map(parent -> TreeUtils.firstAncestorOfKind(parent, Tree.Kind.CALL_EXPR))
      .flatMap(TreeUtils.toStreamInstanceOfMapper(CallExpression.class));
  }

  private boolean hasCleanupCall(FunctionDef function, String filePath) {
    return TreeUtils.firstChild(function, child -> child instanceof CallExpression callExpr && isOsRemoveOrUnlinkCall(callExpr) && hasMatchingPath(callExpr, filePath)).isPresent();
  }

  private boolean isOsRemoveOrUnlinkCall(CallExpression callExpression) {
    return osRemoveType.check(callExpression.callee().typeV2()).isTrue() ||
      osUnlinkType.check(callExpression.callee().typeV2()).isTrue();
  }

  private static boolean hasMatchingPath(CallExpression call, String targetPath) {
    RegularArgument regularArg = TreeUtils.nthArgumentOrKeyword(0, "path", call.arguments());

    if (regularArg == null) {
      return false;
    }

    String pathValue = getStringValue(regularArg.expression());
    return targetPath.equals(pathValue);
  }

  @Nullable
  private static String getStringValue(Expression expression) {
    if (expression instanceof StringLiteral stringLiteral) {
      return stringLiteral.trimmedQuotesValue();
    }
    if (expression instanceof Name name) {
      var assignedValue = Expressions.singleAssignedValue(name);
      if (assignedValue instanceof StringLiteral stringLiteral) {
        return stringLiteral.trimmedQuotesValue();
      }
    }
    return null;
  }

}
