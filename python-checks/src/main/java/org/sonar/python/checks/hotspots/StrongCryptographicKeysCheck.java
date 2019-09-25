/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.checks.hotspots;

import java.util.List;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.SubscriptionContext;
import org.sonar.python.api.tree.HasSymbol;
import org.sonar.python.api.tree.PyArgumentTree;
import org.sonar.python.api.tree.PyCallExpressionTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyNumericLiteralTree;
import org.sonar.python.api.tree.PyQualifiedExpressionTree;
import org.sonar.python.api.tree.Tree.Kind;
import org.sonar.python.semantic.Symbol;

@Rule(key = "S4426")
public class StrongCryptographicKeysCheck extends PythonSubscriptionCheck {

  private static final Pattern CRYPTOGRAPHY = Pattern.compile("cryptography.hazmat.primitives.asymmetric.(rsa|dsa|ec).generate_private_key");
  private static final Pattern CRYPTOGRAPHY_FORBIDDEN_CURVE = Pattern.compile("(SECP192R1|SECT163K1|SECT163R2)");
  private static final Pattern CRYPTO = Pattern.compile("Crypto.PublicKey.(RSA|DSA).generate");
  private static final Pattern CRYPTODOME = Pattern.compile("Cryptodome.PublicKey.(RSA|DSA).generate");


  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, ctx -> {
      PyCallExpressionTree callExpression = (PyCallExpressionTree) ctx.syntaxNode();
      List<PyArgumentTree> arguments = callExpression.arguments();
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


  private static String getQualifiedName(PyCallExpressionTree callExpression) {
    Symbol symbol = callExpression.calleeSymbol();
    return symbol != null && symbol.fullyQualifiedName() != null ? symbol.fullyQualifiedName() : "";
  }

  private abstract static class CryptoAPICheck {
    private static final int CURVE_ARGUMENT_POSITION = 0;

    abstract int getKeySizeArgumentPosition();

    abstract int getExponentArgumentPosition();

    abstract String getKeySizeKeywordName();

    abstract String getExponentKeywordName();

    private boolean isNonCompliantKeySizeArgument(PyArgumentTree argument, int index) {
      PyNameTree keywordArgument = argument.keywordArgument();
      if (keywordArgument == null) {
        return index == getKeySizeArgumentPosition() && isLessThan2048(argument.expression());
      }
      return (keywordArgument).name().equals(getKeySizeKeywordName()) && isLessThan2048(argument.expression());
    }

    private boolean isNonCompliantExponentArgument(PyArgumentTree argument, int index) {
      PyNameTree keywordArgument = argument.keywordArgument();
      if (keywordArgument == null) {
        return index == getExponentArgumentPosition() && isLessThan65537(argument.expression());
      }
      return (keywordArgument).name().equals(getExponentKeywordName()) && isLessThan65537(argument.expression());
    }

    private static boolean isLessThan2048(PyExpressionTree expression) {
      try {
        return expression.is(Kind.NUMERIC_LITERAL) && ((PyNumericLiteralTree) expression).valueAsLong() < 2048;
      } catch (NumberFormatException nfe) {
        return false;
      }
    }

    private static boolean isLessThan65537(PyExpressionTree expression) {
      try {
        return expression.is(Kind.NUMERIC_LITERAL) && ((PyNumericLiteralTree) expression).valueAsLong() < 65537;
      } catch (NumberFormatException nfe) {
        return false;
      }
    }

    private static boolean isNonCompliantCurveArgument(PyArgumentTree argument, int index) {
      PyNameTree keywordArgument = argument.keywordArgument();
      if (keywordArgument == null) {
        return index == CURVE_ARGUMENT_POSITION && isNonCompliantCurve(argument.expression());
      }
      return (keywordArgument).name().equals("curve") && isNonCompliantCurve(argument.expression());
    }

    private static boolean isNonCompliantCurve(PyExpressionTree expression) {
      if (!expression.is(Kind.QUALIFIED_EXPR)) {
        return false;
      }
      PyQualifiedExpressionTree qualifiedExpressionTree = (PyQualifiedExpressionTree) expression;
      if (qualifiedExpressionTree.qualifier() instanceof HasSymbol) {
        Symbol symbol = ((HasSymbol) qualifiedExpressionTree.qualifier()).symbol();
        if (symbol == null || !"cryptography.hazmat.primitives.asymmetric.ec".equals(symbol.fullyQualifiedName())) {
          return false;
        }
        return CRYPTOGRAPHY_FORBIDDEN_CURVE.matcher(qualifiedExpressionTree.name().name()).matches();
      }
      return false;
    }

    void checkArguments(SubscriptionContext ctx, List<PyArgumentTree> arguments) {
      int index = 0;
      for (PyArgumentTree argument : arguments) {
        if (isNonCompliantKeySizeArgument(argument, index)) {
          ctx.addIssue(argument, "Use a key length of at least 2048 bits.");
        }

        if (isNonCompliantExponentArgument(argument, index)) {
          ctx.addIssue(argument, "Use a public key exponent of at least 65537.");
        }

        if (this instanceof CryptographyModuleCheck && isNonCompliantCurveArgument(argument, index)) {
          ctx.addIssue(argument, "Use a key length of at least 224 bits.");
        }
      }
    }

  }

  private static class CryptographyModuleCheck extends CryptoAPICheck {

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
