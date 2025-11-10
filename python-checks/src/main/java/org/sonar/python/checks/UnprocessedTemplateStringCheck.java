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

import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ComprehensionIf;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.tree.WhileStatement;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.StringLiteralImpl;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7942")
public class UnprocessedTemplateStringCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "This template string should be processed before use.";

  private static final List<String> BUILTIN_FUNCTION_NAMES = List.of(
    "print", "str", "int", "float", "bool");

  private static final List<String> LOGGING_METHOD_TYPES = List.of(
    "logging.debug", "logging.info", "logging.warning", "logging.error", "logging.critical");

  private boolean isPython314OrGreater = false;

  private TypeCheckMap<String> builtinFunctionTypeCheckers;
  private TypeCheckMap<String> loggingMethodTypeCheckers;
  private TypeCheckBuilder isStringFormat;
  private TypeCheckBuilder isStringJoin;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initializeState);

    context.registerSyntaxNodeConsumer(Tree.Kind.IF_STMT, ctx -> checkCondition(ctx, ((IfStatement) ctx.syntaxNode()).condition()));
    context.registerSyntaxNodeConsumer(Tree.Kind.WHILE_STMT, ctx -> checkCondition(ctx, ((WhileStatement) ctx.syntaxNode()).condition()));
    context.registerSyntaxNodeConsumer(Tree.Kind.COMP_IF, ctx -> checkCondition(ctx, ((ComprehensionIf) ctx.syntaxNode()).condition()));

    context.registerSyntaxNodeConsumer(Tree.Kind.CONDITIONAL_EXPR, this::checkConditionalExpression);

    context.registerSyntaxNodeConsumer(Tree.Kind.COMPARISON, this::checkBinaryOperation);
    context.registerSyntaxNodeConsumer(Tree.Kind.IN, this::checkBinaryOperation);

    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCallExpression);
  }

  private void initializeState(SubscriptionContext ctx) {
    isPython314OrGreater = PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(ctx.sourcePythonVersions(), PythonVersionUtils.Version.V_314);

    builtinFunctionTypeCheckers = new TypeCheckMap<>();
    for (String functionName : BUILTIN_FUNCTION_NAMES) {
      TypeCheckBuilder typeCheckBuilder = ctx.typeChecker().typeCheckBuilder().isBuiltinWithName(functionName);
      builtinFunctionTypeCheckers.put(typeCheckBuilder, functionName);
    }

    loggingMethodTypeCheckers = new TypeCheckMap<>();
    for (String methodType : LOGGING_METHOD_TYPES) {
      TypeCheckBuilder typeCheckBuilder = ctx.typeChecker().typeCheckBuilder().isTypeWithName(methodType);
      loggingMethodTypeCheckers.put(typeCheckBuilder, methodType);
    }

    isStringFormat = ctx.typeChecker().typeCheckBuilder().isTypeWithName("str.format");
    isStringJoin = ctx.typeChecker().typeCheckBuilder().isTypeWithName("str.join");
  }

  private void checkCondition(SubscriptionContext ctx, Expression condition) {
    if (!isPython314OrGreater) {
      return;
    }
    raiseIfIsUnprocessedTemplateString(ctx, condition);
  }

  private static void raiseIfIsUnprocessedTemplateString(SubscriptionContext ctx, Expression expression) {
    if (isUnprocessedTemplateString(expression)) {
      ctx.addIssue(expression, MESSAGE);
    }
  }

  private static boolean isUnprocessedTemplateString(Expression expression) {
    if (isTemplateString(expression)) {
      return true;
    }

    if (expression instanceof Name name) {
      return Expressions.singleAssignedNonNameValue(name)
        .map(UnprocessedTemplateStringCheck::isTemplateString)
        .orElse(false);
    }

    return false;
  }

  private static boolean isTemplateString(Expression expression) {
    if (expression instanceof StringLiteralImpl stringLiteral) {
      return stringLiteral.isTemplate();
    }
    return false;
  }

  private void checkBinaryOperation(SubscriptionContext ctx) {
    if (!isPython314OrGreater) {
      return;
    }

    BinaryExpression binaryExpr = (BinaryExpression) ctx.syntaxNode();
    raiseIfIsUnprocessedTemplateString(ctx, binaryExpr.leftOperand());
    raiseIfIsUnprocessedTemplateString(ctx, binaryExpr.rightOperand());
  }

  private void checkConditionalExpression(SubscriptionContext ctx) {
    if (!isPython314OrGreater) {
      return;
    }

    ConditionalExpression conditionalExpr = (ConditionalExpression) ctx.syntaxNode();
    raiseIfIsUnprocessedTemplateString(ctx, conditionalExpr.trueExpression());
    raiseIfIsUnprocessedTemplateString(ctx, conditionalExpr.falseExpression());
  }

  private void checkCallExpression(SubscriptionContext ctx) {
    if (!isPython314OrGreater) {
      return;
    }

    CallExpression call = (CallExpression) ctx.syntaxNode();

    PythonType typeToCheck = call.callee().typeV2();
    if (builtinFunctionTypeCheckers.containsForType(typeToCheck) ||
      loggingMethodTypeCheckers.containsForType(typeToCheck) ||
      isStringFormat.check(typeToCheck).isTrue()) {
      call.arguments().forEach(arg -> {
        if (arg instanceof RegularArgument regArg) {
          raiseIfIsUnprocessedTemplateString(ctx, regArg.expression());
        }
      });
      return;
    }

    if (isStringJoin.check(typeToCheck).isTrue()) {
      checkJoinArguments(ctx, call);
    }
  }

  private static void checkJoinArguments(SubscriptionContext ctx, CallExpression call) {
    call.arguments().forEach(arg -> {
      if (arg instanceof RegularArgument regArg) {
        Expression argExpression = regArg.expression();
        if (argExpression instanceof ListLiteral listLiteral) {
          listLiteral.elements().expressions().forEach(element -> raiseIfIsUnprocessedTemplateString(ctx, element));
        } else if (argExpression instanceof Tuple tuple) {
          tuple.elements().forEach(element -> raiseIfIsUnprocessedTemplateString(ctx, element));
        } else {
          // We still raise if there is a template string directly passed to .join
          raiseIfIsUnprocessedTemplateString(ctx, argExpression);
        }
      }
    });
  }
}
