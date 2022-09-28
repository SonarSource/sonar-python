package org.sonar.python.checks.cdk;

import java.util.Locale;
import java.util.Optional;
import java.util.function.Predicate;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.checks.cdk.CdkUtils.getArgument;

public class CdkPredicate {

  private CdkPredicate() {

  }

  public static Predicate<Expression> isFalse() {
    return expression -> Optional.ofNullable(expression.firstToken()).map(Token::value).filter("False"::equals).isPresent();
  }

  public static Predicate<Expression> isNone() {
    return expression -> expression.is(Tree.Kind.NONE);
  }

  public static Predicate<Expression> isFqn(String fqnValue) {
    return expression ->  Optional.ofNullable(TreeUtils.fullyQualifiedNameFromExpression(expression))
      .filter(fqnValue::equals)
      .isPresent();
  }

  /**
   * @return Predicate which tests if expression is a string and is equal the expected value
   */
  public static Predicate<Expression> isStringValue(String expectedValue) {
    return expression -> CdkUtils.getStringValue(expression).filter(expectedValue::equals).isPresent();
  }

  public static Predicate<Expression> isSensitiveMethod(SubscriptionContext ctx, String methodFqn, String argName, Predicate<Expression> sensitiveValuePredicate) {
    return expression -> {
      if (!isFqn(methodFqn).test(expression)) {
        return false;
      }
      if (!expression.is(Tree.Kind.CALL_EXPR)) {
        return true;
      }

      Optional<CdkUtils.ExpressionTrace> argTrace = getArgument(ctx, (CallExpression) expression, argName);
      if (argTrace.isEmpty()) {
        return true;
      }

      return argTrace.filter(trace -> trace.hasExpression(sensitiveValuePredicate)).isPresent();
    };
  }

  public static Predicate<Expression> startsWith(String expected) {
    return e -> CdkUtils.getStringValue(e).filter(str -> str.toLowerCase(Locale.ROOT).startsWith(expected)).isPresent();
  }

}
