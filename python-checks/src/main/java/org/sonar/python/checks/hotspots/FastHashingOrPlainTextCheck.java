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

import java.util.Collection;
import java.util.List;
import java.util.Map;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.hotspots.CommonValidationUtils.ArgumentValidator;

import static org.sonar.python.checks.hotspots.CommonValidationUtils.isLessThan;
import static org.sonar.python.semantic.SymbolUtils.qualifiedNameOrEmpty;

@Rule(key = "S5344")
public class FastHashingOrPlainTextCheck extends PythonSubscriptionCheck {

  private static final String SCRYPT_PARAMETERS_MESSAGE = "Use strong scrypt parameters.";
  private static final ArgumentValidator SCRYPT_R = new ArgumentValidator(
    3, "r", (ctx, argument) -> {
    if (isLessThan(argument.expression(), 8)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });
  private static final ArgumentValidator SCRYPT_BUFLEN = new ArgumentValidator(
    5, "buflen", (ctx, argument) -> {
    if (isLessThan(argument.expression(), 32)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });
  private static final ArgumentValidator SCRYPT_N = new ArgumentValidator(
    2, "N", (ctx, argument) -> {
    if (isLessThan(argument.expression(), (int) Math.pow(2, 13)) || CommonValidationUtils.isLessThanExponent(argument.expression(), 13)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });

  private static final ArgumentValidator HASHLIB_R = new ArgumentValidator(
    3, "r", (ctx, argument) -> {
    if (isLessThan(argument.expression(), 8)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });
  private static final ArgumentValidator HASHLIB_N = new ArgumentValidator(
    2, "n", (ctx, argument) -> {
    if (isLessThan(argument.expression(), (int) Math.pow(2, 13)) || CommonValidationUtils.isLessThanExponent(argument.expression(), 13)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });
  private static final ArgumentValidator HASHLIB_DKLEN = new ArgumentValidator(
    6, "dklen", (ctx, argument) -> {
    if (isLessThan(argument.expression(), 32)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });

  private static final ArgumentValidator CRYPTOGRAPHY_R = new ArgumentValidator(
    3, "r", (ctx, argument) -> {
    if (isLessThan(argument.expression(), 8)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });
  private static final ArgumentValidator CRYPTOGRAPHY_N = new ArgumentValidator(
    2, "n", (ctx, argument) -> {
    if (isLessThan(argument.expression(), (int) Math.pow(2, 13)) || CommonValidationUtils.isLessThanExponent(argument.expression(), 13)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });
  private static final ArgumentValidator CRYPTOGRAPHY_LENGTH = new ArgumentValidator(
    1, "length", (ctx, argument) -> {
    if (isLessThan(argument.expression(), 32)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });

  private static final ArgumentValidator PASSLIB_BLOCK_SIZE = new ArgumentValidator(
    3, "block_size", (ctx, argument) -> {
    if (isLessThan(argument.expression(), 8)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });
  private static final ArgumentValidator PASSLIB_ROUNDS = new ArgumentValidator(
    2, "rounds", (ctx, argument) -> {
    if (isLessThan(argument.expression(), 12)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });


  private static final Map<String, Collection<CommonValidationUtils.CallValidator>> CALL_EXPRESSION_VALIDATORS = Map.of(
    "scrypt.hash", List.of(SCRYPT_R, SCRYPT_BUFLEN, SCRYPT_N),
    "hashlib.scrypt", List.of(HASHLIB_R, HASHLIB_N, HASHLIB_DKLEN),
    "cryptography.hazmat.primitives.kdf.scrypt.Scrypt", List.of(CRYPTOGRAPHY_R, CRYPTOGRAPHY_N, CRYPTOGRAPHY_LENGTH),
    "passlib.hash.scrypt.using", List.of(PASSLIB_BLOCK_SIZE, PASSLIB_ROUNDS)
  );


  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, FastHashingOrPlainTextCheck::checkCallExpr);
  }

  private static void checkCallExpr(SubscriptionContext subscriptionContext) {
    CallExpression callExpression = (CallExpression) subscriptionContext.syntaxNode();
    var qualifiedName = qualifiedNameOrEmpty(callExpression);
    var configs = CALL_EXPRESSION_VALIDATORS.getOrDefault(qualifiedName, List.of());

    configs.forEach(config -> config.validate(subscriptionContext, callExpression));
  }

}
