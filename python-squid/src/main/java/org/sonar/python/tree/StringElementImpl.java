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
package org.sonar.python.tree;

import java.util.Collections;
import java.util.List;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.tree.Tree;

public class StringElementImpl extends PyTree implements StringElement {

  private final String value;
  private final Token token;

  public StringElementImpl(Token token) {
    value = token.value();
    this.token = token;
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
    return Collections.singletonList(token);
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
    return value.substring(0, value.length() - removePrefix(value).length());
  }

  @Override
  public boolean isTripleQuoted() {
    return isTripleQuote(removePrefix(value));
  }

  private static boolean isTripleQuote(String trimmed) {
    if (trimmed.length() >= 6) {
      char startChar = trimmed.charAt(0);
      return startChar == trimmed.charAt(1) && startChar == trimmed.charAt(2);
    }
    return false;
  }

  private static String removePrefix(String value) {
    if (isCharQuote(value.charAt(0))) {
      return value;
    }
    return removePrefix(value.substring(1));
  }

  private static boolean isCharQuote(char character) {
    return character == '\'' || character == '\"';
  }
}
