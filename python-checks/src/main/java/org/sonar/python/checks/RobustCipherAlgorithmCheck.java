/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import java.util.HashSet;
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
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.ReachingValues;
import org.sonar.python.tree.TreeUtils;

// https://jira.sonarsource.com/browse/RSPEC-5547 (general)
// https://jira.sonarsource.com/browse/RSPEC-5552 (python-specific)
@Rule(key = "S5547")
public class RobustCipherAlgorithmCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use a strong cipher algorithm.";
  private static final String INSECURE_CIPHERS_PREFIX = "cryptography.hazmat.decrepit.ciphers.";
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

  public static final Set<String> SSL_SET_CIPHERS_FQN = Set.of(
    "ssl.SSLContext.set_ciphers",
    "OpenSSL.SSL.Context.set_cipher_list"
  );

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

  // The decrepit namespace is intentionally rejected as a whole, including APIs without stubs.
  private static final TypeMatcher INSECURE_CALLEE = TypeMatchers.any(
    TypeMatchers.withFQNPrefix(INSECURE_CIPHERS_PREFIX),
    TypeMatchers.any(SENSITIVE_CALLEE_FQNS.stream().map(fqn -> fqn.startsWith("cryptography.")
      ? TypeMatchers.isType(fqn) : TypeMatchers.withFQN(fqn))));
  private static final TypeMatcher SSL_SET_CIPHERS = TypeMatchers.any(SSL_SET_CIPHERS_FQN.stream().map(TypeMatchers::isType));
  private static final TypeMatcher URLLIB_SSL_CONTEXT = TypeMatchers.isType("urllib3.util.ssl_.create_urllib3_context");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, RobustCipherAlgorithmCheck::checkCallExpression);
  }

  private static void checkCallExpression(SubscriptionContext ctx) {
    var call = (CallExpression) ctx.syntaxNode();
    if (INSECURE_CALLEE.isTrueFor(call.callee(), ctx)) {
      ctx.addIssue(call.callee(), MESSAGE);
    } else if (SSL_SET_CIPHERS.isTrueFor(call.callee(), ctx)) {
      checkForInsecureCiphers(ctx, call);
    } else if (URLLIB_SSL_CONTEXT.isTrueFor(call.callee(), ctx)) {
      Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(4, "ciphers", call.arguments()))
        .map(RegularArgument::expression)
        .ifPresent(expression -> processCiphersArgument(ctx, call, expression));
    }
  }

  private static void checkForInsecureCiphers(SubscriptionContext ctx, CallExpression callExpression) {
    Optional.of(callExpression.arguments())
      .filter(list -> list.size() == 1)
      .map(list -> list.get(0))
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(RegularArgument.class))
      .map(RegularArgument::expression)
      .ifPresent(expression -> processCiphersArgument(ctx, callExpression, expression));
  }

  private static void processCiphersArgument(SubscriptionContext ctx, CallExpression callExpression, Expression expression) {
    StringLiteral stringLiteral = unpackArgument(expression, ctx);
    Optional.ofNullable(stringLiteral)
      .map(StringLiteral::trimmedQuotesValue)
      .map(RobustCipherAlgorithmCheck::findInsecureCiphers)
      .filter(Predicate.not(Set::isEmpty))
      .ifPresent(insecureCiphers -> {
        var secondaryMessage = insecureCiphers.size() > 1 ? "The following cipher strings are insecure: " :
          "The following cipher string is insecure: ";
        secondaryMessage = insecureCiphers.stream().collect(Collectors.joining("`, `", secondaryMessage + "`", "`"));
        ctx.addIssue(callExpression.callee(), MESSAGE)
          .secondary(stringLiteral, secondaryMessage);
      });
  }

  @CheckForNull
  private static StringLiteral unpackArgument(@Nullable Expression expression, SubscriptionContext ctx) {
    var visited = new HashSet<Expression>();
    while (expression instanceof Name name && visited.add(expression)) {
      expression = ReachingValues.singleValue(name, ctx);
    }
    return expression instanceof StringLiteral literal ? literal : null;
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

