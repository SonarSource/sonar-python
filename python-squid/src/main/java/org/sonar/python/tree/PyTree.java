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
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.PyToken;
import org.sonar.python.api.tree.Tree;

public abstract class PyTree implements Tree {
  private final AstNode node;
  private final PyToken firstToken;
  private final PyToken lastToken;
  private Tree parent = null;

  public PyTree(@Nullable AstNode node) {
    this.node = node;
    this.firstToken = node == null ? null : new PyTokenImpl(node.getToken());
    this.lastToken = node == null ? null : new PyTokenImpl(node.getLastToken());
  }

  public PyTree(PyToken firstToken, PyToken lastToken) {
    this.node = null;
    this.firstToken = firstToken;
    this.lastToken = lastToken;
  }

  @Override
  public boolean is(Kind kind) {
    return kind == getKind();
  }

  @Override
  @CheckForNull
  public AstNode astNode() {
    return node;
  }

  @Override
  public PyToken firstToken() {
    return firstToken;
  }

  @Override
  public PyToken lastToken() {
    return lastToken;
  }

  @Override
  public Tree parent() {
    return parent;
  }

  protected void setParent(Tree parent) {
    this.parent = parent;
  }
}
