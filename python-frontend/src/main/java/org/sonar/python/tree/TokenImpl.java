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

import com.sonar.sslr.api.TokenType;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.PythonLine;
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
  private boolean isCompressed = false;

  public TokenImpl(com.sonar.sslr.api.Token token) {
    this.token = token;
    this.trivia = token.getTrivia().stream().map(tr -> new TriviaImpl(new TokenImpl(tr.getToken()))).collect(Collectors.toList());
  }

  public TokenImpl(com.sonar.sslr.api.Token token, int line, int column, int includedEscapeChars, List<Trivia> trivia, boolean isCompressed) {
    this.token = token;
    this.line = line;
    this.column = column;
    this.includedEscapeChars = includedEscapeChars;
    this.trivia = trivia;
    this.isCompressed = isCompressed;
  }

  @Override
  public String value() {
    return token.getValue();
  }

  @Override
  public int line() {
    return line != null ? line : pythonLine().line();
  }

  @Override
  public int column() {
    return column != null ? column : pythonColumn();
  }

  @Override
  public PythonLine pythonLine() {
    return new PythonLine(token.getLine());
  }

  @Override
  public int includedEscapeChars() {
    return includedEscapeChars;
  }

  @Override
  public int pythonColumn() {
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

  @Override
  public boolean isCompressed() {
    return this.isCompressed;
  }

}
