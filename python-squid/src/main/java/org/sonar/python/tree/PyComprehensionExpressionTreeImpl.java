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
import java.util.Arrays;
import java.util.List;
import org.sonar.python.api.tree.PyComprehensionForTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyComprehensionExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyComprehensionExpressionTreeImpl extends PyTree implements PyComprehensionExpressionTree {

  private final Kind kind;
  private final PyExpressionTree resultExpression;
  private final PyComprehensionForTree comprehensionFor;

  public PyComprehensionExpressionTreeImpl(Kind kind, PyToken openingToken, PyExpressionTree resultExpression,
                                           PyComprehensionForTree compFor, PyToken closingToken) {
    super(openingToken, closingToken);
    this.kind = kind;
    this.resultExpression = resultExpression;
    this.comprehensionFor = compFor;
  }

  @Override
  public PyExpressionTree resultExpression() {
    return resultExpression;
  }

  @Override
  public PyComprehensionForTree comprehensionFor() {
    return comprehensionFor;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitPyListOrSetCompExpression(this);
  }

  @Override
  public List<Tree> children() {
    return Arrays.asList(resultExpression, comprehensionFor);
  }

  @Override
  public Kind getKind() {
    return kind;
  }
}
