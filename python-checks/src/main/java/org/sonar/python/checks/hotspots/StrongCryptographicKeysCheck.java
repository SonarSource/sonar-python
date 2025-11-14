/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
import java.util.Set;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.checks.hotspots.CommonValidationUtils.ArgumentValidator;
import org.sonar.python.checks.hotspots.CommonValidationUtils.CallValidator;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.semantic.SymbolUtils.qualifiedNameOrEmpty;

@Rule(key = "S4426")
public class StrongCryptographicKeysCheck extends PythonSubscriptionCheck {
  public static final String MESSAGE_AT_LEAST_65537_EXPONENT = "Use a public key exponent of at least 65537.";
  public static final String MESSAGE_AT_LEAST_2048_BIT = "Use a key length of at least 2048 bits.";
  public static final String MESSAGE_AT_LEAST_224_BIT = "Use a key length of at least 224 bits.";
  public static final String MESSAGE_NIST_APPROVED_CURVE = "Use a NIST-approved elliptic curve.";
  public static final String CURVE = "curve";
  private static final Pattern CRYPTOGRAPHY_FORBIDDEN_CURVE = Pattern.compile("(SECP192R1|SECT571K1|SECT409K1|SECT283K1|SECT233K1|SECT163K1|SECT571R1|SECT409R1|SECT283R1|SECT233R1|SECT163R2)");
  private static final ArgumentValidator CRYPTOGRAPHY_KEY_SIZE = new ArgumentValidator(
    1, "key_size", (ctx, argument) -> {
    if (isLessThan2048(argument)) {
      ctx.addIssue(argument, MESSAGE_AT_LEAST_2048_BIT);
    }
  });

  private static final ArgumentValidator CRYPTOGRAPHY_PUBLIC_EXPONENT = new ArgumentValidator(
    0, "public_exponent", (ctx, argument) -> {
    if (isLessThan65537(argument)) {
      ctx.addIssue(argument, MESSAGE_AT_LEAST_65537_EXPONENT);
    }
  });

  private static final ArgumentValidator CRYPTOGRAPHY_CURVE = new ArgumentValidator(
    0, CURVE, (ctx, argument) -> {
    if (isNonCompliantCurve(argument.expression())) {
      ctx.addIssue(argument, MESSAGE_AT_LEAST_224_BIT);
    }
  });

  private static final ArgumentValidator CRYPTO_CRYPTODOME_KEY_SIZE = new ArgumentValidator(
    0, "bits", (ctx, argument) -> {
    if (isLessThan2048(argument)) {
      ctx.addIssue(argument, MESSAGE_AT_LEAST_2048_BIT);
    }
  });

  private static final ArgumentValidator CRYPTO_EXPONENT = new ArgumentValidator(
    3, "e", (ctx, argument) -> {
    if (isLessThan65537(argument)) {
      ctx.addIssue(argument, MESSAGE_AT_LEAST_65537_EXPONENT);
    }
  });

  private static final ArgumentValidator CRYPTODOME_EXPONENT = new ArgumentValidator(
    2, "e", (ctx, argument) -> {
    if (isLessThan65537(argument)) {
      ctx.addIssue(argument, MESSAGE_AT_LEAST_65537_EXPONENT);
    }
  });

  private static final ArgumentValidator CRYPTODOME_ELGAMAL_CURVE = new ArgumentValidator(
    0, CURVE, (ctx, argument) -> {
    if (isLessThan2048(argument)) {
      ctx.addIssue(argument, MESSAGE_AT_LEAST_2048_BIT);
    }
  });

  private static final ArgumentValidator CRYPTODOME_ECC_FORBIDDEN_CURVE = new ArgumentValidator(
    0, CURVE, (ctx, argument) -> {
    if (isForbiddenCurve(argument.expression(), Set.of("NIST P-192", "p192", "P-192", "prime192v1", "secp192r1"))) {
      ctx.addIssue(argument, MESSAGE_NIST_APPROVED_CURVE);
    }
  });

  private static final Map<String, Collection<CallValidator>> CALL_EXPRESSION_VALIDATORS =
    Map.ofEntries(
      Map.entry("cryptography.hazmat.primitives.asymmetric.rsa.generate_private_key", List.of(CRYPTOGRAPHY_KEY_SIZE, CRYPTOGRAPHY_PUBLIC_EXPONENT, CRYPTOGRAPHY_CURVE)),
      Map.entry("cryptography.hazmat.primitives.asymmetric.dsa.generate_private_key", List.of(CRYPTOGRAPHY_KEY_SIZE, CRYPTOGRAPHY_PUBLIC_EXPONENT, CRYPTOGRAPHY_CURVE)),
      Map.entry("cryptography.hazmat.primitives.asymmetric.ec.generate_private_key", List.of(CRYPTOGRAPHY_KEY_SIZE, CRYPTOGRAPHY_PUBLIC_EXPONENT, CRYPTOGRAPHY_CURVE)),
      Map.entry("cryptography.hazmat.primitives.asymmetric.dh.generate_parameters", List.of(CRYPTOGRAPHY_KEY_SIZE)),
      Map.entry("Crypto.PublicKey.RSA.generate", List.of(CRYPTO_CRYPTODOME_KEY_SIZE, CRYPTO_EXPONENT)),
      Map.entry("Crypto.PublicKey.DSA.generate", List.of(CRYPTO_CRYPTODOME_KEY_SIZE, CRYPTO_EXPONENT)),
      Map.entry("Cryptodome.PublicKey.RSA.generate", List.of(CRYPTO_CRYPTODOME_KEY_SIZE, CRYPTODOME_EXPONENT)),
      Map.entry("Cryptodome.PublicKey.DSA.generate", List.of(CRYPTO_CRYPTODOME_KEY_SIZE, CRYPTODOME_EXPONENT)),
      Map.entry("Cryptodome.PublicKey.ElGamal.generate", List.of(CRYPTODOME_ELGAMAL_CURVE)),
      Map.entry("Cryptodome.PublicKey.ECC.generate", List.of(CRYPTODOME_ECC_FORBIDDEN_CURVE)),
      Map.entry("OpenSSL.crypto.PKey.generate_key", List.of(new OpenSslCryptoModuleCallValidator()))
    );

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      var qualifiedName = qualifiedNameOrEmpty(callExpression);
      var configs = CALL_EXPRESSION_VALIDATORS.getOrDefault(qualifiedName, List.of());

      configs.forEach(config -> config.validate(ctx, callExpression));
    });
  }


  private static class OpenSslCryptoModuleCallValidator implements CallValidator {
    private static final Set<String> KEY_TYPE_FQNS_TO_CHECK = Set.of("OpenSSL.crypto.TYPE_RSA", "OpenSSL.crypto.TYPE_DSA");

    @Override
    public void validate(SubscriptionContext ctx, CallExpression callExpression) {
      var arguments = callExpression.arguments();
      if (keyTypeNeedsToBeChecked(arguments)) {
        TreeUtils.nthArgumentOrKeywordOptional(1, "bits", arguments)
          .filter(StrongCryptographicKeysCheck::isLessThan2048)
          .ifPresent(arg -> ctx.addIssue(arg, MESSAGE_AT_LEAST_2048_BIT));
      }
    }

    private static boolean keyTypeNeedsToBeChecked(List<Argument> arguments) {
      return TreeUtils.nthArgumentOrKeywordOptional(0, "type", arguments)
        .map(RegularArgument::expression)
        .filter(HasSymbol.class::isInstance)
        .map(HasSymbol.class::cast)
        .map(HasSymbol::symbol)
        .map(Symbol::fullyQualifiedName)
        .map(KEY_TYPE_FQNS_TO_CHECK::contains)
        .orElse(false);
    }
  }

  private static boolean isForbiddenCurve(Expression expression, Set<String> forbiddenStrings) {
    if (expression.getKind() == Kind.STRING_LITERAL) {
      String curveName = ((StringLiteral) expression).trimmedQuotesValue();
      return forbiddenStrings.contains(curveName);
    }
    if (expression.getKind() == Kind.NAME) {
      return Expressions.singleAssignedNonNameValue(((Name) expression))
        .map(v -> isForbiddenCurve(v, forbiddenStrings))
        .orElse(false);
    }
    return false;
  }

  private static boolean isNonCompliantCurve(Expression expression) {
    if (expression instanceof Name name) {
      expression = Expressions.singleAssignedValue(name);
      if (expression == null) {
        return false;
      }
    }
    if (expression instanceof CallExpression callExpression) {
      expression = callExpression.callee();
    }
    if (!expression.is(Kind.QUALIFIED_EXPR)) {
      return false;
    }
    QualifiedExpression qualifiedExpressionTree = (QualifiedExpression) expression;
    if (qualifiedExpressionTree.qualifier() instanceof HasSymbol hasSymbol) {
      Symbol symbol = hasSymbol.symbol();
      if (symbol == null || !"cryptography.hazmat.primitives.asymmetric.ec".equals(symbol.fullyQualifiedName())) {
        return false;
      }
      return CRYPTOGRAPHY_FORBIDDEN_CURVE.matcher(qualifiedExpressionTree.name().name()).matches();
    }
    return false;
  }

  private static boolean isLessThan2048(RegularArgument argument) {
    return CommonValidationUtils.isLessThan(argument.expression(), 2048);
  }

  private static boolean isLessThan65537(RegularArgument argument) {
    return CommonValidationUtils.isLessThan(argument.expression(), 65537);
  }

}
