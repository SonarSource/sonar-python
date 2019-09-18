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
import java.util.Collections;
import java.util.List;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.PyYieldExpressionTree;
import org.sonar.python.api.tree.Tree;

public class PyYieldExpressionTreeImpl extends PyTree implements PyYieldExpressionTree {
  private final PyToken yieldKeyword;
  private final PyToken fromKeyword;
  private final List<PyExpressionTree> expressionTrees;

  public PyYieldExpressionTreeImpl(AstNode astNode, PyToken yieldKeyword, @Nullable PyToken fromKeyword, List<PyExpressionTree> expressionTrees) {
    super(astNode);
    this.yieldKeyword = yieldKeyword;
    this.fromKeyword = fromKeyword;
    this.expressionTrees = expressionTrees;
  }

  @Override
  public PyToken yieldKeyword() {
    return yieldKeyword;
  }

  @CheckForNull
  @Override
  public PyToken fromKeyword() {
    return fromKeyword;
  }

  @Override
  public List<PyExpressionTree> expressions() {
    return expressionTrees;
  }

  @Override
  public Kind getKind() {
    return Kind.YIELD_EXPR;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitYieldExpression(this);
  }

  @Override
  public List<Tree> children() {
    return Collections.unmodifiableList(expressionTrees);
  }
}
