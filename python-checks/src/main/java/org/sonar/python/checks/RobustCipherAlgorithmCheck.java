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

import java.util.LinkedHashSet;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.StringLiteralImpl;
import org.sonar.python.tree.TreeUtils;

// https://jira.sonarsource.com/browse/RSPEC-5547 (general)
// https://jira.sonarsource.com/browse/RSPEC-5552 (python-specific)
@Rule(key = "S5547")
public class RobustCipherAlgorithmCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use a strong cipher algorithm.";
  private static final Set<String> INSECURE_CIPHERS_PREFIXES = Set.of("cryptography.hazmat.decrepit.ciphers.");
  private static final Set<String> INSECURE_CIPHERS = Set.of(
    "NULL",
    "aNULL",
    "eNULL",
    "COMPLEMENTOFALL",
    "RC2",
    "RC4",
    "IDEA",
    "SEED",
    "DES",
    "3DES",
    "MD5",
    "SHA",
    "SHA1",
    "ADH",
    "AECDH",
    "CBC",
    "LOW",
    "@SECLEVEL=0",
    "@SECLEVEL=1",
    "DEFAULT@SECLEVEL=0",
    "DEFAULT@SECLEVEL=1"
  );

  public static final String SSL_SET_CIPHERS_FQN = "ssl.SSLContext.set_ciphers";

  private static final Set<String> SENSITIVE_CALLEE_FQNS = Set.of(
    "Crypto.Cipher.ARC2.new",
    "Crypto.Cipher.ARC4.new",
    "Crypto.Cipher.Blowfish.new",
    "Crypto.Cipher.XOR.new",
    "Crypto.Cipher.CAST.new",
    "Crypto.Cipher.DES.new",
    "Crypto.Cipher.DES3.new",
    "Cryptodome.Cipher.ARC2.new",
    "Cryptodome.Cipher.ARC4.new",
    "Cryptodome.Cipher.Blowfish.new",
    "Cryptodome.Cipher.XOR.new",
    "Cryptodome.Cipher.CAST.new",
    "Cryptodome.Cipher.DES.new",
    "Cryptodome.Cipher.DES3.new",
    "cryptography.hazmat.primitives.ciphers.algorithms.ARC4",
    "cryptography.hazmat.primitives.ciphers.algorithms.Blowfish",
    "cryptography.hazmat.primitives.ciphers.algorithms.IDEA",
    "cryptography.hazmat.primitives.ciphers.algorithms.CAST5",
    "cryptography.hazmat.primitives.ciphers.algorithms.TripleDES",
    "pyDes.des",
    "pyDes.triple_des"
  );



  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, RobustCipherAlgorithmCheck::checkCallExpression);
  }

  private static void checkCallExpression(SubscriptionContext subscriptionContext) {
    CallExpression callExpr = (CallExpression) subscriptionContext.syntaxNode();
    Optional.of(callExpr)
      .map(CallExpression::calleeSymbol)
      .map(Symbol::fullyQualifiedName)
      .ifPresent(fullyQualifiedName -> {
        if (SENSITIVE_CALLEE_FQNS.contains(fullyQualifiedName) || INSECURE_CIPHERS_PREFIXES.stream().anyMatch(fullyQualifiedName::startsWith)) {
          subscriptionContext.addIssue(callExpr.callee(), MESSAGE);
        } else if (SSL_SET_CIPHERS_FQN.equals(fullyQualifiedName)) {
          checkForInsecureCiphers(subscriptionContext, callExpr);
        }
      });
  }

  private static void checkForInsecureCiphers(SubscriptionContext ctx, CallExpression callExpression) {
    Optional.of(callExpression.arguments())
      .filter(list -> list.size() == 1)
      .map(list -> list.get(0))
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(RegularArgument.class))
      .map(RegularArgument::expression)
      .map(RobustCipherAlgorithmCheck::unpackArgument)
      .ifPresent(stringLiteral -> Optional.of(stringLiteral.trimmedQuotesValue())
        .map(RobustCipherAlgorithmCheck::findInsecureCiphers)
        .filter(Predicate.not(Set::isEmpty))
        .ifPresent(insecureCiphers -> {
          var secondaryMessage = insecureCiphers.size() > 1 ? "The following cipher strings are insecure: " :
            "The following cipher string is insecure: ";
          secondaryMessage = insecureCiphers.stream().collect(Collectors.joining("`, `", secondaryMessage + "`", "`"));

          ctx.addIssue(callExpression.callee(), MESSAGE)
            .secondary(stringLiteral, secondaryMessage);
        }));
  }

  @CheckForNull
  private static StringLiteral unpackArgument(@Nullable Expression expression) {
    if (expression == null) {
      return null;
    } else if (expression.is(Tree.Kind.STRING_LITERAL)) {
      return ((StringLiteralImpl) expression);
    } else if (expression.is(Tree.Kind.NAME)) {
      return unpackArgument(Expressions.singleAssignedValue((Name) expression));
    } else {
      return null;
    }
  }

  private static Set<String> findInsecureCiphers(String ciphers) {
    return Stream.of(ciphers)
      .flatMap(str -> Stream.of(str.split(":")))
      .filter(str -> !str.startsWith("!") && !str.startsWith("-"))
      .flatMap(str -> Stream.of(str.split("\\+")))
      .flatMap(str -> Stream.of(str.split("-")))
      .filter(INSECURE_CIPHERS::contains)
      .collect(Collectors.toCollection(LinkedHashSet::new));
  }
}

