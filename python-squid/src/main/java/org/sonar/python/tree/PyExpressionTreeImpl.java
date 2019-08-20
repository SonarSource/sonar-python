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
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;

// FIXME : this class is a placeholder while we implement concrete type of expressions and should be deleted.
public class PyExpressionTreeImpl extends PyTree implements PyExpressionTree {
  public PyExpressionTreeImpl(AstNode astNode) {
    super(astNode);
  }

  @Override
  public Kind getKind() {
    return null;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    //TODO : remove
  }
}
