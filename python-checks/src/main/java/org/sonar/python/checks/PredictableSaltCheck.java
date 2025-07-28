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
import java.util.List;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S2053")
public class PredictableSaltCheck extends PythonSubscriptionCheck {

  private static final String MISSING_SALT_MESSAGE = "Add an unpredictable salt value to this hash.";
  private static final String PREDICTABLE_SALT_MESSAGE = "Make this salt unpredictable.";
  private static final String DIFFERENT_SALT_THAN_KEY_MATERIAL_MESSAGE = "Make this salt different than the derived key material.";
  private static final String SALT_IS_USED_HERE_MESSAGE = "The salt is used in the derive method here.";
  private static final String SALT_ARGUMENT_NAME = "salt";
  private static final String PASSWORD_ARGUMENT_NAME = "password";

  private static final Map<String, ArgumentInfo> SENSITIVE_ARGUMENT_BY_FQN = Map.ofEntries(
    Map.entry("hashlib.pbkdf2_hmac", new ArgumentInfo(2, SALT_ARGUMENT_NAME, new ArgumentInfo(1, PASSWORD_ARGUMENT_NAME))),
    Map.entry("hashlib.scrypt", new ArgumentInfo(4, SALT_ARGUMENT_NAME, new ArgumentInfo(0, PASSWORD_ARGUMENT_NAME))),
    Map.entry("crypt.crypt", new ArgumentInfo(1, SALT_ARGUMENT_NAME)),
    Map.entry("cryptography.hazmat.primitives.kdf.pbkdf2.PBKDF2HMAC", new ArgumentInfo(2, SALT_ARGUMENT_NAME)),
    Map.entry("cryptography.hazmat.primitives.kdf.scrypt.Scrypt", new ArgumentInfo(0, SALT_ARGUMENT_NAME)),
    Map.entry("Cryptodome.Protocol.KDF.PBKDF2", new ArgumentInfo(1, SALT_ARGUMENT_NAME, new ArgumentInfo(0, PASSWORD_ARGUMENT_NAME))),
    Map.entry("Cryptodome.Protocol.KDF.scrypt", new ArgumentInfo(1, SALT_ARGUMENT_NAME, new ArgumentInfo(0, PASSWORD_ARGUMENT_NAME))),
    Map.entry("Cryptodome.Protocol.KDF.bcrypt", new ArgumentInfo(2, SALT_ARGUMENT_NAME, new ArgumentInfo(0, PASSWORD_ARGUMENT_NAME))),
    Map.entry("Crypto.Protocol.KDF.PBKDF2", new ArgumentInfo(1, SALT_ARGUMENT_NAME, new ArgumentInfo(0, PASSWORD_ARGUMENT_NAME))),
    Map.entry("Crypto.Protocol.KDF.scrypt", new ArgumentInfo(1, SALT_ARGUMENT_NAME, new ArgumentInfo(0, PASSWORD_ARGUMENT_NAME))),
    Map.entry("Crypto.Protocol.KDF.bcrypt", new ArgumentInfo(2, SALT_ARGUMENT_NAME, false, new ArgumentInfo(0, PASSWORD_ARGUMENT_NAME))));

  private static final List<String> SENSITIVE_DERIVE_FUNCTIONS_FQN = List.of(
    "cryptography.hazmat.primitives.kdf.pbkdf2.PBKDF2HMAC.derive",
    "cryptography.hazmat.primitives.kdf.scrypt.Scrypt.derive");

  private static final Map<String, ArgumentInfo> SALT_FUNCTION_ARGUMENTS_TO_CHECK = Map.of(
    "bytes.fromhex", new ArgumentInfo(0, "__string"),
    "bytearray.fromhex", new ArgumentInfo(0, "__string"),
    "base64.b64decode", new ArgumentInfo(0, "s"),
    "base64.b64encode", new ArgumentInfo(0, "s"),
    "base64.b32encode", new ArgumentInfo(0, "s"),
    "base64.b32decode", new ArgumentInfo(0, "s"),
    "base64.b16encode", new ArgumentInfo(0, "s"),
    "base64.b16decode", new ArgumentInfo(0, "s"));

  private TypeCheckMap<ArgumentInfo> sensitiveArgumentByFqnCheck;
  private TypeCheckMap<ArgumentInfo> saltFunctionArgumentsToCheck;
  private List<TypeCheckBuilder> deriveFunctionsToCheck;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      sensitiveArgumentByFqnCheck = new TypeCheckMap<>();
      saltFunctionArgumentsToCheck = new TypeCheckMap<>();
      deriveFunctionsToCheck = new ArrayList<>();
      initializeTypeChecks(ctx, sensitiveArgumentByFqnCheck,
        saltFunctionArgumentsToCheck, deriveFunctionsToCheck);
    });
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> handleCallExpression((CallExpression) ctx.syntaxNode(),
      ctx,
      sensitiveArgumentByFqnCheck,
      saltFunctionArgumentsToCheck,
      deriveFunctionsToCheck));
  }

  private static void initializeTypeChecks(SubscriptionContext ctx,
    TypeCheckMap<ArgumentInfo> sensitiveArgumentByFqnCheck,
    TypeCheckMap<ArgumentInfo> saltFunctionArgumentsToCheck, List<TypeCheckBuilder> deriveFunctionsToCheck) {
    SENSITIVE_ARGUMENT_BY_FQN.forEach((fqn, argumentNumber) -> {
      var checker = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(fqn);
      sensitiveArgumentByFqnCheck.put(checker, argumentNumber);
    });
    SALT_FUNCTION_ARGUMENTS_TO_CHECK.forEach((fqn, argumentInfo) -> {
      var checker = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(fqn);
      saltFunctionArgumentsToCheck.put(checker, argumentInfo);
    });
    SENSITIVE_DERIVE_FUNCTIONS_FQN.forEach(fqn -> {
      var checker = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(fqn);
      deriveFunctionsToCheck.add(checker);
    });
  }

  private static void handleCallExpression(CallExpression callExpression, SubscriptionContext ctx,
    TypeCheckMap<ArgumentInfo> sensitiveArgumentByFqnCheck, TypeCheckMap<ArgumentInfo> saltFunctionArgumentsToCheck, List<TypeCheckBuilder> deriveFunctionsToCheck) {
    Optional.of(callExpression)
      .map(CallExpression::callee)
      .map(Expression::typeV2)
      .map(sensitiveArgumentByFqnCheck::getForType)
      .ifPresent(argumentInfo -> checkArguments(callExpression, argumentInfo, ctx, saltFunctionArgumentsToCheck, deriveFunctionsToCheck));
  }

  private static void checkArguments(CallExpression callExpression, ArgumentInfo argumentInfo, SubscriptionContext ctx,
    TypeCheckMap<ArgumentInfo> saltFunctionArgumentsToCheck, List<TypeCheckBuilder> deriveFunctionsToCheck) {
    var argument = TreeUtils.nthArgumentOrKeyword(argumentInfo.position(), argumentInfo.name(), callExpression.arguments());
    if (argument != null) {
      if (hasRaisedOnSensitiveArgument(argument, ctx, saltFunctionArgumentsToCheck)) {
        return;
      }
      if (hasRaisedOnSameArgument(argument, argumentInfo, callExpression, ctx)) {
        return;
      }
      if (hasRaisedOnSameSaltAndDerivedKeyMaterial(argument, deriveFunctionsToCheck, ctx)) {
        return;
      }

    } else if (argumentInfo.required() && callExpression.arguments().stream().noneMatch(UnpackingExpression.class::isInstance)) {
      ctx.addIssue(callExpression.callee(), MISSING_SALT_MESSAGE);
    }
  }

  private static boolean hasRaisedOnSameSaltAndDerivedKeyMaterial(RegularArgument argument, List<TypeCheckBuilder> deriveFunctionsToCheck, SubscriptionContext ctx) {
    if (!argument.expression().is(Tree.Kind.NAME)) {
      return false;
    }
    var symbol = ((Name) argument.expression()).symbolV2();
    if (symbol == null) {
      return false;
    }
    var usagesInDeriveCall = symbol.usages().stream()
      .map(usage -> usage.tree().parent())
      .flatMap(TreeUtils.toStreamInstanceOfMapper(RegularArgument.class))
      .filter(regArg -> !regArg.equals(argument))
      .filter(regArg -> isUsedInDeriveCall(regArg, deriveFunctionsToCheck))
      .map(RegularArgument::expression)
      .toList();

    if (usagesInDeriveCall.isEmpty()) {
      return false;
    }
    var issue = ctx.addIssue(argument, DIFFERENT_SALT_THAN_KEY_MATERIAL_MESSAGE);
    usagesInDeriveCall.stream().forEach(arg -> issue.secondary(arg, SALT_IS_USED_HERE_MESSAGE));
    return true;
  }

  private static boolean isUsedInDeriveCall(RegularArgument regularArgument, List<TypeCheckBuilder> deriveFunctionsToCheck) {
    return Optional.ofNullable(regularArgument.parent())
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(ArgList.class))
      .map(ArgList::parent)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
      .map(CallExpression::callee)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .map(qualifiedExpr -> deriveFunctionsToCheck.stream()
        .anyMatch(deriveFunction -> deriveFunction.check(qualifiedExpr.name().typeV2()) == TriBool.TRUE))
      .orElse(false);

  }

  private static boolean hasRaisedOnSameArgument(RegularArgument argument, ArgumentInfo argumentInfo, CallExpression callExpression, SubscriptionContext ctx) {
    return Optional.ofNullable(argumentInfo.shouldNotBeSameAsArgument())
      .map(ai -> TreeUtils.nthArgumentOrKeyword(ai.position(), ai.name(), callExpression.arguments()))
      .map(shouldNotBeSameAsArgument -> raisedOnSameArgument(argument, shouldNotBeSameAsArgument, ctx))
      .orElseGet(() -> false);

  }

  private static boolean raisedOnSameArgument(RegularArgument argument, RegularArgument shouldNotBeSameAsArgument, SubscriptionContext ctx) {
    var exp1 = argument.expression();
    var exp2 = shouldNotBeSameAsArgument.expression();
    if (exp1 instanceof Name n1
      && exp2 instanceof Name n2
      && n1.symbolV2() == n2.symbolV2()) {
      ctx.addIssue(argument, PREDICTABLE_SALT_MESSAGE).secondary(shouldNotBeSameAsArgument, "");
      return true;
    }
    return false;
  }

  private static boolean hasRaisedOnSensitiveArgument(RegularArgument regularArgument,
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
        return true;
      } else {
        expression = null;
      }
    }
    return false;
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
      .map(argumentInfo -> Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(argumentInfo.position(), argumentInfo.name(),
        callExpression.arguments()))
        .map(RegularArgument::expression)
        .orElse(expression))
      .orElse(null);
  }

  private record ArgumentInfo(int position, String name, boolean required, @Nullable ArgumentInfo shouldNotBeSameAsArgument) {
    private ArgumentInfo(int position, String name) {
      this(position, name, true, null);
    }

    private ArgumentInfo(int position, String name, ArgumentInfo shouldNotBeSameAsArgument) {
      this(position, name, true, shouldNotBeSameAsArgument);
    }
  }

}
