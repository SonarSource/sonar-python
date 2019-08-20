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
import org.sonar.python.api.tree.Tree;

public abstract class PyTree extends AstNode implements Tree {
  private final AstNode node;

  public PyTree(AstNode node) {
    super(node.getType(), node.getName(), node.getToken());
    this.node = node;
    for (AstNode child : node.getChildren()) {
      addChild(child);
    }
  }

  public abstract Kind getKind();

  @Override
  public boolean is(Kind kind) {
    return kind == getKind();
  }

  @Override
  public AstNode astNode() {
    return node;
  }
}
