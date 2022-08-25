package org.sonar.python.checks.tests;

import java.util.Optional;
import java.util.Set;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.python.tree.TreeUtils;

public class PytestUtils {
  private static final String LIB_NAME = "pytest";
  private static final Set<String> raiseMethods = Set.of("raises");

  private PytestUtils() {}

  public static boolean isARaiseCall(QualifiedExpression qualifiedExpression) {
    return isPytest(qualifiedExpression) && raiseMethods.contains(qualifiedExpression.name().name());
  }

  public static boolean isPytest(QualifiedExpression qualifiedExpression) {
    return TreeUtils.getSymbolFromTree(qualifiedExpression.qualifier())
      .stream()
      .anyMatch(symbol -> LIB_NAME.equals(symbol.fullyQualifiedName()));
  }
}
