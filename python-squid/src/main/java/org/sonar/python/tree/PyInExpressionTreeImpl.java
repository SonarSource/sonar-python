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

import org.sonar.python.api.tree.PyToken;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyInExpressionTree;

public class PyInExpressionTreeImpl extends PyBinaryExpressionTreeImpl implements PyInExpressionTree {

  private final PyToken notToken;

  public PyInExpressionTreeImpl(PyExpressionTree leftOperand, @Nullable PyToken not, PyToken operator, PyExpressionTree rightOperand) {
    super(leftOperand, operator, rightOperand);
    this.notToken = not;
  }

  @Override
  public Kind getKind() {
    return Kind.IN;
  }

  @CheckForNull
  @Override
  public PyToken notToken() {
    return notToken;
  }
}
