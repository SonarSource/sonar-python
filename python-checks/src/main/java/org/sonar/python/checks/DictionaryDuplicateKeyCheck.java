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
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.DictionaryLiteralElement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;

@Rule(key = "S5780")
public class DictionaryDuplicateKeyCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.DICTIONARY_LITERAL, ctx -> {
      DictionaryLiteral dictionaryLiteral = (DictionaryLiteral) ctx.syntaxNode();
      Set<Integer> issueIndexes = new HashSet<>();
      if (dictionaryLiteral.elements().size() > 100) {
        return;
      }
      for (int i = 0; i < dictionaryLiteral.elements().size(); i++) {
        if (!dictionaryLiteral.elements().get(i).is(Tree.Kind.KEY_VALUE_PAIR) || issueIndexes.contains(i)) {
          continue;
        }
        Expression key = ((KeyValuePair) dictionaryLiteral.elements().get(i)).key();
        List<Tree> duplicateKeys = findIdenticalKeys(i, dictionaryLiteral.elements(), issueIndexes);
        if (!duplicateKeys.isEmpty()) {
          PreciseIssue issue = ctx.addIssue(key, "Change or remove duplicates of this key.");
          duplicateKeys.forEach(d -> issue.secondary(d, "Duplicate key"));
        }
      }
    });
  }

  private List<Tree> findIdenticalKeys(int startIndex, List<DictionaryLiteralElement> elements, Set<Integer> issueIndexes) {
    Expression key = ((KeyValuePair) elements.get(startIndex)).key();
    List<Tree> duplicates = new ArrayList<>();
    for (int i = startIndex + 1; i < elements.size(); i++) {
      if (!elements.get(i).is(Tree.Kind.KEY_VALUE_PAIR)) {
        continue;
      }
      Expression comparedKey = ((KeyValuePair) elements.get(i)).key();
      if (isSameKey(key, comparedKey)) {
        issueIndexes.add(i);
        duplicates.add(comparedKey);
      }
    }
    return duplicates;
  }

  private boolean isSameKey(Expression key, Expression comparedKey) {
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
    // BigDecimal#compareTo is required as equals() returns true only with identical scales
    return toBigDecimal(key).compareTo(toBigDecimal(comparedKey)) == 0;
  }

  private BigDecimal toBigDecimal(Tree numberTree) {
    if (numberTree.is(Tree.Kind.NUMERIC_LITERAL)) {
      return parseAsBigDecimal(((NumericLiteral) numberTree).valueAsString());
    }
    return ((Name) numberTree).name().equals("True") ? BigDecimal.ONE : BigDecimal.ZERO;
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

  public BigDecimal parseAsBigDecimal(String numberLiteralValue) {
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
