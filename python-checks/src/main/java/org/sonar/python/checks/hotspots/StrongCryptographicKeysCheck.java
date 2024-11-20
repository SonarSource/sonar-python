/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.List;
import java.util.Optional;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S4426")
public class StrongCryptographicKeysCheck extends PythonSubscriptionCheck {

  private static final Pattern CRYPTOGRAPHY = Pattern.compile("cryptography.hazmat.primitives.asymmetric.(rsa|dsa|ec).generate_private_key");
  private static final Pattern CRYPTOGRAPHY_FORBIDDEN_CURVE = Pattern.compile("(SECP192R1|SECT163K1|SECT163R2)");
  private static final Pattern CRYPTO = Pattern.compile("Crypto.PublicKey.(RSA|DSA).generate");
  private static final Pattern CRYPTODOME = Pattern.compile("Cryptodome.PublicKey.(RSA|DSA).generate");


  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      List<Argument> arguments = callExpression.arguments();
      String qualifiedName = getQualifiedName(callExpression);
      if (CRYPTOGRAPHY.matcher(qualifiedName).matches()) {
        new CryptographyModuleCheck().checkArguments(ctx, arguments);
      } else if (CRYPTO.matcher(qualifiedName).matches()) {
        new CryptoModuleCheck().checkArguments(ctx, arguments);
      } else if (CRYPTODOME.matcher(qualifiedName).matches()) {
        new CryptodomeModuleCheck().checkArguments(ctx, arguments);
      }
    });
  }


  private static String getQualifiedName(CallExpression callExpression) {
    Symbol symbol = callExpression.calleeSymbol();
    return symbol != null && symbol.fullyQualifiedName() != null ? symbol.fullyQualifiedName() : "";
  }

  private abstract static class CryptoAPICheck {

    abstract int getKeySizeArgumentPosition();

    abstract int getExponentArgumentPosition();

    abstract String getKeySizeKeywordName();

    abstract String getExponentKeywordName();

    private static boolean isLessThan2048(RegularArgument argument) {
      return isLessThan(argument.expression(), 2048);
    }

    private static boolean isLessThan65537(RegularArgument argument) {
      return isLessThan(argument.expression(), 65537);
    }

    private static boolean isLessThan(Expression expression, int number) {
      try {
        return expression.is(Kind.NUMERIC_LITERAL) && ((NumericLiteral) expression).valueAsLong() < number;
      } catch (NumberFormatException nfe) {
        return false;
      }
    }

    void checkArguments(SubscriptionContext ctx, List<Argument> arguments) {
      argument(getKeySizeArgumentPosition(), getKeySizeKeywordName(), arguments)
        .filter(CryptoAPICheck::isLessThan2048)
        .ifPresent(arg -> ctx.addIssue(arg, "Use a key length of at least 2048 bits."));

      argument(getExponentArgumentPosition(), getExponentKeywordName(), arguments)
        .filter(CryptoAPICheck::isLessThan65537)
        .ifPresent(arg -> ctx.addIssue(arg, "Use a public key exponent of at least 65537."));
    }
  }

  public static Optional<RegularArgument> argument(int argPosition, String keyword, List<Argument> arguments) {
    return Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(argPosition, keyword, arguments));
  }

  private static class CryptographyModuleCheck extends CryptoAPICheck {

    private static final int CURVE_ARGUMENT_POSITION = 0;

    @Override
    protected int getKeySizeArgumentPosition() {
      return 1;
    }

    @Override
    protected int getExponentArgumentPosition() {
      return 0;
    }

    @Override
    protected String getKeySizeKeywordName() {
      return "key_size";
    }

    @Override
    protected String getExponentKeywordName() {
      return "public_exponent";
    }

    @Override
    void checkArguments(SubscriptionContext ctx, List<Argument> arguments) {
      super.checkArguments(ctx, arguments);

      argument(CURVE_ARGUMENT_POSITION, "curve", arguments)
        .filter(arg -> isNonCompliantCurve(arg.expression()))
        .ifPresent(arg -> ctx.addIssue(arg, "Use a key length of at least 224 bits."));
    }

    private static boolean isNonCompliantCurve(Expression expression) {
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
  }

  private static class CryptoModuleCheck extends CryptoAPICheck {

    @Override
    protected int getExponentArgumentPosition() {
      return 3;
    }

    @Override
    protected int getKeySizeArgumentPosition() {
      return 0;
    }

    @Override
    protected String getExponentKeywordName() {
      return "e";
    }

    @Override
    protected String getKeySizeKeywordName() {
      return "bits";
    }
  }

  private static class CryptodomeModuleCheck extends CryptoAPICheck {

    @Override
    protected int getExponentArgumentPosition() {
      return 2;
    }

    @Override
    protected String getExponentKeywordName() {
      return "e";
    }

    @Override
    protected String getKeySizeKeywordName() {
      return "bits";
    }

    @Override
    protected int getKeySizeArgumentPosition() {
      return 0;
    }
  }

}
