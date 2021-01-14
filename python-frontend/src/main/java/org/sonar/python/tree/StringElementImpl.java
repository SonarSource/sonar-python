/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.python.tree;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FormattedExpression;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class StringElementImpl extends PyTree implements StringElement {

  private final String value;
  private final Token token;
  private List<FormattedExpression> formattedExpressions = new ArrayList<>();

  public StringElementImpl(Token token) {
    value = token.value();
    this.token = token;
  }

  @Override
  public Token firstToken() {
    return token;
  }

  @Override
  public Token lastToken() {
    return token;
  }

  @Override
  public Kind getKind() {
    return Kind.STRING_ELEMENT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitStringElement(this);
  }

  @Override
  public List<Tree> computeChildren() {
    // Warning: in the case of f-strings, there's a kind of overlap between `token` and `formattedExpressions`: they
    // are different representations of the same analyzed code.
    // TreeUtils.tokens() doesn't contain the tokens of the formattedExpressions.
    return Stream.concat(Stream.of(token), formattedExpressions.stream()).collect(Collectors.toList());
  }

  @Override
  public String value() {
    return value;
  }

  @Override
  public String trimmedQuotesValue() {
    String trimmed = removePrefix(value);
    // determine if string is using long string or short string format
    int startIndex = 1;
    if (isTripleQuote(trimmed)) {
      startIndex = 3;
    }
    return trimmed.substring(startIndex, trimmed.length() - startIndex);
  }

  @Override
  public String prefix() {
    return value.substring(0, prefixLength(value));
  }

  @Override
  public boolean isTripleQuoted() {
    return isTripleQuote(removePrefix(value));
  }

  @Override
  public boolean isInterpolated() {
    String prefix = prefix();
    return prefix.indexOf('f') >= 0 || prefix.indexOf('F') >= 0;
  }

  @Override
  public List<Expression> interpolatedExpressions() {
    return formattedExpressions.stream().map(FormattedExpression::expression).collect(Collectors.toList());
  }

  @Override
  public List<FormattedExpression> formattedExpressions() {
    return formattedExpressions;
  }

  void addFormattedExpression(FormattedExpression formattedExpression) {
    formattedExpressions.add(formattedExpression);
  }


  private static boolean isTripleQuote(String trimmed) {
    if (trimmed.length() >= 6) {
      char startChar = trimmed.charAt(0);
      return startChar == trimmed.charAt(1) && startChar == trimmed.charAt(2);
    }
    return false;
  }

  private static String removePrefix(String value) {
    return value.substring(prefixLength(value));
  }

  private static boolean isCharQuote(char character) {
    return character == '\'' || character == '\"';
  }

  private static int prefixLength(String value) {
    int prefixLength = 0;
    while (!isCharQuote(value.charAt(prefixLength))) {
      prefixLength++;
    }
    return prefixLength;
  }

  public int contentStartIndex() {
    int prefixLength = prefixLength(value);
    if (isTripleQuote(value.substring(prefixLength))) {
      return prefixLength + 3;
    }
    return prefixLength + 1;
  }
}
