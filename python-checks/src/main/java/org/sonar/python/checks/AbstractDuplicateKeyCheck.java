/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.python.checks;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;

public abstract class AbstractDuplicateKeyCheck extends PythonSubscriptionCheck {

  protected void reportDuplicates(List<Expression> trees, SubscriptionContext ctx, String message, String messageSecondary) {
    Map<Value, List<Tree>> valuesToTrees = trees.stream().collect(Collectors.groupingBy(AbstractDuplicateKeyCheck::evaluate));
    for (Map.Entry<Value, List<Tree>> valueTrees : valuesToTrees.entrySet()) {
      List<Tree> treesEvaluatingToSameValue = valueTrees.getValue();
      if (treesEvaluatingToSameValue.size() > 1) {
        PreciseIssue issue = ctx.addIssue(treesEvaluatingToSameValue.get(0), message);
        treesEvaluatingToSameValue.stream().skip(1).forEach(duplicate -> issue.secondary(duplicate, messageSecondary));
      }
    }
  }

  /** ADT marker interface for normalized, easily hashable & comparable values. */
  interface Value {}

  /** Value-implementing wrapper for BigDecimals and boolean values (which can be equal to numbers in python). */
  static class NumericValue implements Value {
    private final BigDecimal wrapped;
    private final int hashCode;
    public NumericValue(BigDecimal wrapped) {
      this.wrapped = wrapped.stripTrailingZeros();
      this.hashCode = this.wrapped.hashCode();
    }

    @Override
    public int hashCode() {
      return hashCode;
    }

    @Override
    public boolean equals(Object other) {
      if (other == this) {
        return true;
      } else if (other == null) {
        return false;
      } else if (other instanceof NumericValue) {
        return this.hashCode == ((NumericValue) other).hashCode &&
          Objects.equals(((NumericValue) other).wrapped, this.wrapped);
      } else {
        return false;
      }
    }
  }

  /** Value-implementing wrapper for Strings. */
  static class StringValue implements Value {
    private final String trimmedQuotesValue;
    private final String withPrefixes;
    private final int hashCode;
    public StringValue(String trimmedQuotesValue, String withPrefixes){
      this.trimmedQuotesValue = trimmedQuotesValue;
      this.withPrefixes = withPrefixes;
      this.hashCode = Objects.hash(trimmedQuotesValue, withPrefixes);
    }

    @Override
    public int hashCode() {
      return hashCode;
    }

    @Override
    public boolean equals(Object other) {
      if (other == this) {
        return true;
      } else if (other == null) {
        return false;
      } else if (other instanceof StringValue) {
        StringValue otherStringValue = (StringValue) other;
        return this.hashCode == otherStringValue.hashCode &&
          Objects.equals(this.trimmedQuotesValue, otherStringValue.trimmedQuotesValue) &&
          Objects.equals(this.withPrefixes, otherStringValue.withPrefixes);
      } else {
        return false;
      }
    }
  }

  /**
   * Partially evaluated syntactic trees with other values as child nodes.
   *
   * Similar to <code>PyTree</code>, but with evaluated strings and numeric literals at the leaves.
   */
  static class TreeValue implements Value {
    private final Tree.Kind kind;
    private List<Value> children;
    private final int hashCode;
    public TreeValue(Tree.Kind kind, List<Value> children) {
      this.kind = kind;
      this.children = Collections.unmodifiableList(children);
      hashCode = Objects.hash(kind, children);
    }

    @Override
    public int hashCode() {
      return hashCode;
    }

    @Override
    public boolean equals(Object other) {
      if (other == this) {
        return true;
      } else if (other == null) {
        return false;
      } else if (other instanceof TreeValue) {
        TreeValue otherTreeValue = (TreeValue) other;
        return this.hashCode == otherTreeValue.hashCode &&
          this.kind != Tree.Kind.CALL_EXPR &&
          Objects.equals(this.kind, otherTreeValue.kind) &&
          Objects.equals(this.children, otherTreeValue.children);
      } else {
        return false;
      }
    }
  }

  /** Unevaluated syntactic leaf nodes. */
  static class TokenValue implements Value {
    private final Tree.Kind kind;
    private final String wrapped;
    private final int hashCode;
    public TokenValue(Tree.Kind kind, String wrapped) {
      this.kind = kind;
      this.wrapped = wrapped;
      hashCode = Objects.hash(kind, wrapped);
    }

    @Override
    public int hashCode() {
      return hashCode;
    }

    @Override
    public boolean equals(Object other) {
      if (other == this) {
        return true;
      } else if (other == null) {
        return false;
      } else if (other instanceof TokenValue) {
        TokenValue otherTreeValue = (TokenValue) other;
        return this.hashCode == otherTreeValue.hashCode &&
          Objects.equals(this.kind, otherTreeValue.kind) &&
          Objects.equals(this.wrapped, otherTreeValue.wrapped);
      } else {
        return false;
      }
    }
  }

  /**
   * Partially evaluates the tree by interpreting nested primitive value literals, and returns easily
   * hashable / equality-comparable values.
   */
  static Value evaluate(Tree tree) {
    if (isANumber(tree)) {
      return new NumericValue(toBigDecimal(tree));
    } else if (tree.is(Tree.Kind.STRING_LITERAL)) {
      StringLiteral strLit = (StringLiteral) tree;
      if (strLit.stringElements().stream().noneMatch(StringElement::isInterpolated)) {
        String trimmedQuotesValue = strLit.trimmedQuotesValue();
        String withPrefixes = strLit.stringElements().stream()
          .map(s -> s.prefix().toLowerCase(Locale.ENGLISH) + s.trimmedQuotesValue()).collect(Collectors.joining());
        return new StringValue(trimmedQuotesValue, withPrefixes);
      }
    } else if (tree.children().isEmpty()) {
      return new TokenValue(tree.getKind(), tree.firstToken().value());
    }
    List<Value> evaluatedChildNodes =
      tree.children().stream().map(AbstractDuplicateKeyCheck::evaluate).collect(Collectors.toList());
    return new TreeValue(tree.getKind(), evaluatedChildNodes);
  }

  private static BigDecimal toBigDecimal(Tree numberTree) {
    if (numberTree.is(Tree.Kind.NUMERIC_LITERAL)) {
      return parseAsBigDecimal(((NumericLiteral) numberTree).valueAsString());
    }
    return ((Name) numberTree).name().equals("True") ? BigDecimal.ONE : BigDecimal.ZERO;
  }

  private static BigDecimal parseAsBigDecimal(String numberLiteralValue) {
    String numberValue = numberLiteralValue.replace("_", "");
    if (numberValue.startsWith("0b") || numberValue.startsWith("0B")) {
      return new BigDecimal(new BigInteger(numberValue.substring(2), 2));
    }
    if (numberValue.startsWith("0o") || numberValue.startsWith("0O")) {
      return new BigDecimal(new BigInteger(numberValue.substring(2), 8));
    }
    if (numberValue.startsWith("0x") || numberValue.startsWith("0X")) {
      return new BigDecimal(new BigInteger(numberValue.substring(2), 16));
    }
    return new BigDecimal(numberValue);
  }

  private static boolean isANumber(Tree tree) {
    return tree.is(Tree.Kind.NUMERIC_LITERAL) || (tree.is(Tree.Kind.NAME) && (((Name) tree).name().equals("True") || ((Name) tree).name().equals("False")));
  }
}
