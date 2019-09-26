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

import com.sonar.sslr.api.TokenType;
import com.sonar.sslr.api.Trivia;
import java.util.Collections;
import java.util.List;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.TreeVisitor;
import org.sonar.python.api.tree.Tree;

public class TokenImpl extends PyTree implements Token {

  private com.sonar.sslr.api.Token token;

  public TokenImpl(com.sonar.sslr.api.Token token) {
    super(null);
    this.token = token;
  }

  public com.sonar.sslr.api.Token token() {
    return token;
  }

  @Override
  public String value() {
    return token.getValue();
  }

  @Override
  public int line() {
    return token.getLine();
  }

  @Override
  public int column() {
    return token.getColumn();
  }

  @Override
  public List<Trivia> trivia() {
    return token.getTrivia();
  }

  public TokenType type() {
    return token.getType();
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitToken(this);
  }

  @Override
  public List<Tree> children() {
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
}
