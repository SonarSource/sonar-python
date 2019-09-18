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
import org.sonar.python.api.tree.PyToken;
import java.util.List;

public abstract class PyDictOrSetLiteralTreeImpl extends PyTree {
  private final PyToken lCurlyBrace;
  private final List<PyToken> commas;
  private final PyToken rCurlyBrace;

  public PyDictOrSetLiteralTreeImpl(AstNode node, PyToken lCurlyBrace, List<PyToken> commas, PyToken rCurlyBrace) {
    super(node);
    this.lCurlyBrace = lCurlyBrace;
    this.commas = commas;
    this.rCurlyBrace = rCurlyBrace;
  }

  public PyToken lCurlyBrace() {
    return lCurlyBrace;
  }

  public PyToken rCurlyBrace() {
    return rCurlyBrace;
  }

  public List<PyToken> commas() {
    return commas;
  }
}
