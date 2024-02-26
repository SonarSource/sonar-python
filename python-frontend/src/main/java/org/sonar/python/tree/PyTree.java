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

import java.util.List;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;

public abstract class PyTree implements Tree {
  protected Token firstToken;
  protected Token lastToken;
  private List<Tree> childs;
  private Tree parent = null;

  protected PyTree() {
  }

  @Override
  public boolean is(Kind... kinds) {
    Kind treeKind = getKind();
    for (Kind kindIter : kinds) {
      if (treeKind == kindIter) {
        return true;
      }
    }
    return false;
  }

  @Override
  public Token firstToken() {
    if (firstToken == null) {
      List<Tree> children = children();
      if (children.isEmpty()) {
        this.firstToken = null;
      } else {
        Tree first = children.get(0);
        this.firstToken = first.is(Kind.TOKEN) ? (Token) first : first.firstToken();
      }
    }
    return firstToken;
  }

  @Override
  public Token lastToken() {
    if (lastToken == null) {
      List<Tree> children = children();
      if (children.isEmpty()) {
        this.firstToken = null;
      } else {
        Tree last = children.get(children.size() - 1);
        this.lastToken = last.is(Kind.TOKEN) ? (Token) last : last.lastToken();
      }
    }
    return lastToken;
  }


  @Override
  public Tree parent() {
    return parent;
  }

  protected void setParent(Tree parent) {
    this.parent = parent;
  }

  abstract List<Tree> computeChildren();

  public List<Tree> children() {
    if (childs == null) {
      childs = computeChildren();
    }
    return childs;
  }
}
