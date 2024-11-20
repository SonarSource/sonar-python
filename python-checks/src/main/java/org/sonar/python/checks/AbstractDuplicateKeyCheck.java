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
package org.sonar.python.checks;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.python.checks.utils.CheckUtils;

public abstract class AbstractDuplicateKeyCheck extends PythonSubscriptionCheck {

  // Avoid performance issues for big dictionary/set literals
  static final int SIZE_THRESHOLD = 100;

  boolean isSameKey(Expression key, Expression comparedKey) {
    if (key.is(Tree.Kind.TUPLE) && comparedKey.is(Tree.Kind.TUPLE)) {
      return areEquivalentTuples((Tuple) key, (Tuple) comparedKey);
    }
    if (key.is(Tree.Kind.STRING_LITERAL) && comparedKey.is(Tree.Kind.STRING_LITERAL)) {
      return areEquivalentStringLiterals((StringLiteral) key, (StringLiteral) comparedKey);
    }
    if (isANumber(key) && isANumber(comparedKey)) {
      return areEquivalentNumbers(key, comparedKey);
    }
    return !key.is(Tree.Kind.CALL_EXPR) && CheckUtils.areEquivalent(key, comparedKey);
  }

  private boolean areEquivalentTuples(Tuple key, Tuple comparedKey) {
    List<Expression> first = key.elements();
    List<Expression> second = comparedKey.elements();
    if (first.size() != second.size()) {
      return false;
    }
    for (int i = 0; i < first.size(); i++) {
      if (!isSameKey(first.get(i), second.get(i))) {
        return false;
      }
    }
    return true;
  }

  private boolean areEquivalentNumbers(Tree key, Tree comparedKey) {
    return toNumber(key).isEquivalentNumber((toNumber(comparedKey)));
  }

  private static Number toNumber(Tree numberTree) {
    if (numberTree.is(Tree.Kind.NUMERIC_LITERAL)) {
      return Number.fromString(((NumericLiteral) numberTree).valueAsString());
    }
    return "True".equals(((Name) numberTree).name()) ? new Number(BigDecimal.ONE) : new Number(BigDecimal.ZERO);
  }

  private static boolean areEquivalentStringLiterals(StringLiteral key, StringLiteral comparedKey) {
    if (key.stringElements().stream().anyMatch(StringElement::isInterpolated) || comparedKey.stringElements().stream().anyMatch(StringElement::isInterpolated)) {
      return false;
    }
    if (key.trimmedQuotesValue().equals(comparedKey.trimmedQuotesValue())) {
      String keyWithPrefixes = key.stringElements().stream()
        .map(s -> s.prefix().toLowerCase(Locale.ENGLISH) + s.trimmedQuotesValue()).collect(Collectors.joining());
      String comparedKeyWithPrefixes = comparedKey.stringElements().stream()
        .map(s -> s.prefix().toLowerCase(Locale.ENGLISH) + s.trimmedQuotesValue()).collect(Collectors.joining());
      return keyWithPrefixes.equals(comparedKeyWithPrefixes);
    }
    return false;
  }

  private static boolean isANumber(Tree tree) {
    return tree.is(Tree.Kind.NUMERIC_LITERAL)
      || (tree.is(Tree.Kind.NAME) && ("True".equals(((Name) tree).name()) || "False".equals(((Name) tree).name())));
  }

  static class Number {
    private final boolean isComplex;
    private final BigDecimal value;

    public Number(BigDecimal value, boolean isComplex) {
      this.value = value;
      this.isComplex = isComplex;
    }

    public Number(BigDecimal value) {
      this.value = value;
      this.isComplex = false;
    }

    public static Number fromString(String str) {
      String numberValue = str.replace("_", "");
      if (numberValue.endsWith("L") || numberValue.endsWith("l")) {
        numberValue = numberValue.substring(0, numberValue.length() - 1);
      }
      if (numberValue.startsWith("0b") || numberValue.startsWith("0B")) {
        return new Number(new BigDecimal(new BigInteger(numberValue.substring(2), 2)));
      }
      if (numberValue.startsWith("0o") || numberValue.startsWith("0O")) {
        return new Number(new BigDecimal(new BigInteger(numberValue.substring(2), 8)));
      }
      if (numberValue.startsWith("0x") || numberValue.startsWith("0X")) {
        return new Number(new BigDecimal(new BigInteger(numberValue.substring(2), 16)));
      }
      if (numberValue.endsWith("j") || numberValue.endsWith("J")) {
        return new Number(new BigDecimal(new BigInteger(numberValue.substring(0, numberValue.length() - 1))), true);
      }
      return new Number(new BigDecimal(numberValue));
    }

    public boolean isEquivalentNumber(Number other) {
      // BigDecimal#compareTo is required as equals() returns true only with identical scales
      if (other.value.compareTo(BigDecimal.ZERO) == 0 && value.compareTo(BigDecimal.ZERO) == 0) {
        return true;
      }
      if (other.isComplex != isComplex) {
        return false;
      }
      return other.value.compareTo(value) == 0;
    }
  }
}
