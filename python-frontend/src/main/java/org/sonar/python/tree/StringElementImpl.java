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
package org.sonar.python.tree;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import javax.annotation.Nullable;

import org.sonar.plugins.python.api.tree.FormattedExpression;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.python.api.PythonTokenType;

public class StringElementImpl extends PyTree implements StringElement {

  private final String value;
  private final Token token;
  private List<Tree> fStringMiddles = new ArrayList<>();

  private final Token fstringEnd;

  public StringElementImpl(Token token, List<Tree> fStringMiddles, @Nullable Token fstringEnd) {
    this.token = token;
    this.fstringEnd = fstringEnd;
    this.fStringMiddles = fStringMiddles;
    value = computeValue(token, fstringEnd, fStringMiddles);
  }

  @Override
  public Token firstToken() {
    return token;
  }

  @Override
  public Token lastToken() {
    return fstringEnd != null ? fstringEnd : token;
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
    List<Tree> children = new ArrayList<>();
    children.add(token);
    children.addAll(fStringMiddles);
    children.add(fstringEnd);
    return children.stream()
      .filter(Objects::nonNull)
      .toList();
  }

  @Override
  public String value() {
    return value;
  }

  @Override
  public String trimmedQuotesValue() {
    if (token.type() == PythonTokenType.FSTRING_MIDDLE) {
      return value;
    }
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
  public List<FormattedExpression> formattedExpressions() {
    return fStringMiddles.stream().filter(FormattedExpression.class::isInstance).map(FormattedExpression.class::cast).toList();
  }

  private static boolean isTripleQuote(String trimmed) {
    if (trimmed.length() >= 6) {
      char startChar = trimmed.charAt(0);
      return startChar == trimmed.charAt(1) && startChar == trimmed.charAt(2);
    }
    return false;
  }

  private String removePrefix(String value) {
    return value.substring(prefixLength(value));
  }

  private static boolean isCharQuote(char character) {
    return character == '\'' || character == '\"';
  }

  private int prefixLength(String value) {
    int prefixLength = 0;
    if (token.type() == PythonTokenType.FSTRING_MIDDLE) {
      return 0;
    }
    while (!isCharQuote(value.charAt(prefixLength))) {
      prefixLength++;
    }
    return prefixLength;
  }

  public int contentStartIndex() {
    int prefixLength = prefixLength(value);
    if (token.type() == PythonTokenType.FSTRING_MIDDLE) {
      return 0;
    }
    if (isTripleQuote(value.substring(prefixLength))) {
      return prefixLength + 3;
    }
    return prefixLength + 1;
  }

  private static final String computeValue(Token token, @Nullable Token fstringEnd, List<Tree> fStringMiddles) {
    String stringContent = fStringMiddles.stream()
      .map(exp -> TreeUtils.treeToString(exp, false))
      .filter(Objects::nonNull)
      .collect(Collectors.joining());
    String end = fstringEnd == null ? "" : fstringEnd.value();
    return String.join("", token.value(), stringContent, end);
  }
}
