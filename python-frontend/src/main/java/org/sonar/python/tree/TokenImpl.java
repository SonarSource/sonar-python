/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import com.sonar.sslr.api.TokenType;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.tree.Trivia;

public class TokenImpl extends PyTree implements Token {

  private com.sonar.sslr.api.Token token;
  private List<Trivia> trivia;

  private Integer line;
  private Integer column;
  private int includedEscapeChars;

  public TokenImpl(com.sonar.sslr.api.Token token) {
    this.token = token;
    this.trivia = token.getTrivia().stream().map(tr -> new TriviaImpl(new TokenImpl(tr.getToken()))).collect(Collectors.toList());
  }

  public TokenImpl(com.sonar.sslr.api.Token token, int line, int column, int includedEscapeChars, List<Trivia> trivia) {
    this.token = token;
    this.line = line;
    this.column = column;
    this.includedEscapeChars = includedEscapeChars;
    this.trivia = trivia;
  }

  @Override
  public String value() {
    return token.getValue();
  }

  @Override
  public int line() {
    return line != null ? line : physicalLine();
  }

  @Override
  public int column() {
    return column != null ? column : physicalColumn();
  }

  @Override
  public int physicalLine() {
    return token.getLine();
  }

  @Override
  public int includedEscapeChars() {
    return includedEscapeChars;
  }

  @Override
  public int physicalColumn() {
    return token.getColumn();
  }

  @Override
  public List<Trivia> trivia() {
    return trivia;
  }

  public TokenType type() {
    return token.getType();
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitToken(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Collections.emptyList();
  }

  @Override
  public Kind getKind() {
    return Tree.Kind.TOKEN;
  }

  @Override
  public Token firstToken() {
    return this;
  }

  @Override
  public Token lastToken() {
    return this;
  }

  @Override
  public int valueLength() {
    return value().length() + includedEscapeChars();
  }
}
