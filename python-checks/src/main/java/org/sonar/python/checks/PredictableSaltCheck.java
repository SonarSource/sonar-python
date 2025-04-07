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

import java.util.ArrayList;
import java.util.Map;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S2053")
public class PredictableSaltCheck extends PythonSubscriptionCheck {

  private static final String MISSING_SALT_MESSAGE = "Add an unpredictable salt value to this hash.";
  private static final String PREDICTABLE_SALT_MESSAGE = "Make this salt unpredictable.";
  private static final Map<String, Integer> SENSITIVE_ARGUMENT_BY_FQN = Map.ofEntries(
    Map.entry("hashlib.pbkdf2_hmac", 2),
    Map.entry("hashlib.scrypt", 4),
    Map.entry("crypt.crypt", 1),
    Map.entry("cryptography.hazmat.primitives.kdf.pbkdf2.PBKDF2HMAC", 2),
    Map.entry("Cryptodome.Protocol.KDF.PBKDF2", 1),
    Map.entry("Cryptodome.Protocol.KDF.scrypt", 1),
    Map.entry("Cryptodome.Protocol.KDF.bcrypt", 2)
  );

  private static final Map<String, ArgumentInfo> SALT_FUNCTION_ARGUMENTS_TO_CHECK = Map.of(
    "bytes.fromhex", new ArgumentInfo(0, "__string"),
    "bytearray.fromhex", new ArgumentInfo(0, "__string"),
    "base64.b64decode", new ArgumentInfo(0, "s"),
    "base64.b64encode", new ArgumentInfo(0, "s"),
    "base64.b32encode", new ArgumentInfo(0, "s"),
    "base64.b32decode", new ArgumentInfo(0, "s"),
    "base64.b16encode", new ArgumentInfo(0, "s"),
    "base64.b16decode", new ArgumentInfo(0, "s")
  );

  @Override
  public void initialize(Context context) {
    var sensitiveArgumentByFqnCheck = new TypeCheckMap<Integer>();
    var saltFunctionArgumentsToCheck = new TypeCheckMap<ArgumentInfo>();
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> initializeTypeChecks(ctx, sensitiveArgumentByFqnCheck,
      saltFunctionArgumentsToCheck));
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> handleCallExpression((CallExpression) ctx.syntaxNode(),
      ctx,
      sensitiveArgumentByFqnCheck,
      saltFunctionArgumentsToCheck
    ));
  }

  private static void initializeTypeChecks(SubscriptionContext ctx,
    TypeCheckMap<Integer> sensitiveArgumentByFqnCheck,
    TypeCheckMap<ArgumentInfo> saltFunctionArgumentsToCheck) {
    SENSITIVE_ARGUMENT_BY_FQN.forEach((fqn, argumentNumber) -> {
      var checker = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(fqn);
      sensitiveArgumentByFqnCheck.put(checker, argumentNumber);
    });
    SALT_FUNCTION_ARGUMENTS_TO_CHECK.forEach((fqn, argumentInfo) -> {
      var checker = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(fqn);
      saltFunctionArgumentsToCheck.put(checker, argumentInfo);
    });
  }

  private static void handleCallExpression(CallExpression callExpression, SubscriptionContext ctx,
    TypeCheckMap<Integer> sensitiveArgumentByFqnCheck, TypeCheckMap<ArgumentInfo> saltFunctionArgumentsToCheck) {
    Optional.of(callExpression)
      .map(CallExpression::callee)
      .map(Expression::typeV2)
      .map(sensitiveArgumentByFqnCheck::getForType)
      .ifPresent(sensitiveArgumentNumber -> checkArguments(callExpression, sensitiveArgumentNumber, ctx, saltFunctionArgumentsToCheck));
  }

  private static void checkArguments(CallExpression callExpression, int argNb, SubscriptionContext ctx,
    TypeCheckMap<ArgumentInfo> saltFunctionArgumentsToCheck) {
    var argument = TreeUtils.nthArgumentOrKeyword(argNb, "salt", callExpression.arguments());
    if (argument != null) {
      checkSensitiveArgument(argument, ctx, saltFunctionArgumentsToCheck);
    } else if (callExpression.arguments().stream().noneMatch(UnpackingExpression.class::isInstance)){
      ctx.addIssue(callExpression.callee(), MISSING_SALT_MESSAGE);
    }
  }

  private static void checkSensitiveArgument(RegularArgument regularArgument,
    SubscriptionContext ctx, TypeCheckMap<ArgumentInfo> saltFunctionArgumentsToCheck) {
    var secondaries = new ArrayList<Tree>();
    var expression = regularArgument.expression();
    while (expression != null) {
      if (expression instanceof Name name) {
        expression = getNameAssignedValueToCheck(name, secondaries);
      } else if (expression instanceof CallExpression callExpression) {
        expression = getCallExpressionArgumentValueToCheck(saltFunctionArgumentsToCheck, callExpression, expression);
      } else if (expression instanceof StringLiteral) {
        var issue = ctx.addIssue(regularArgument, PREDICTABLE_SALT_MESSAGE);
        secondaries.forEach(t -> issue.secondary(t, ""));
        return;
      } else {
        expression = null;
      }
    }
  }

  private static Expression getNameAssignedValueToCheck(Name name, ArrayList<Tree> secondaries) {
    var expression = Expressions.singleAssignedValue(name);
    if (expression != null) {
      var assignmentStatement = TreeUtils.firstAncestorOfKind(expression, Tree.Kind.ASSIGNMENT_STMT);
      secondaries.add(assignmentStatement);
    }
    return expression;
  }

  private static Expression getCallExpressionArgumentValueToCheck(TypeCheckMap<ArgumentInfo> saltFunctionArgumentsToCheck,
    CallExpression callExpression, Expression expression) {
    return Optional.of(callExpression)
      .map(CallExpression::callee)
      .map(Expression::typeV2)
      .map(saltFunctionArgumentsToCheck::getForType)
      .map(argumentInfo -> Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(argumentInfo.position(), argumentInfo.name(), callExpression.arguments()))
        .map(RegularArgument::expression)
        .orElse(expression))
      .orElse(null);
  }

  private record ArgumentInfo(int position, String name) {
  }


}
