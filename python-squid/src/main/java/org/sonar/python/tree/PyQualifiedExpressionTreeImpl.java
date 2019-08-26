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
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyQualifiedExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;

public class PyQualifiedExpressionTreeImpl extends PyExpressionTreeImpl implements PyQualifiedExpressionTree {
  private final PyNameTree name;
  private final PyExpressionTree qualifier;

  public PyQualifiedExpressionTreeImpl(AstNode astNode, PyNameTree name, PyExpressionTree qualifier) {
    super(astNode);
    this.name = name;
    this.qualifier = qualifier;
  }

  @Override
  public PyExpressionTree qualifier() {
    return qualifier;
  }

  @Override
  public PyNameTree name() {
    return name;
  }

  @Override
  public Kind getKind() {
    return Kind.QUALIFIED_EXPR;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitQualifiedExpression(this);
  }
}
