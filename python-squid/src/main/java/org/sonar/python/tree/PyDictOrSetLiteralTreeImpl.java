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

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Token;
import java.util.List;
import org.sonar.python.api.tree.PyDictOrSetLiteralTree;

public abstract class PyDictOrSetLiteralTreeImpl extends PyTree implements PyDictOrSetLiteralTree {
  private final Token lCurlyBrace;
  private final List<Token> commas;
  private final Token rCurlyBrace;

  public PyDictOrSetLiteralTreeImpl(AstNode node, Token lCurlyBrace, List<Token> commas, Token rCurlyBrace) {
    super(node);
    this.lCurlyBrace = lCurlyBrace;
    this.commas = commas;
    this.rCurlyBrace = rCurlyBrace;
  }

  @Override
  public Token lCurlyBrace() {
    return lCurlyBrace;
  }

  @Override
  public Token rCurlyBrace() {
    return rCurlyBrace;
  }

  @Override
  public List<Token> commas() {
    return commas;
  }
}
