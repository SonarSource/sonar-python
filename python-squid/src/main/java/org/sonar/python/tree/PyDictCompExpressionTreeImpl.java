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
import org.sonar.python.api.tree.PyDictCompExpressionTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyDictCompExpressionTreeImpl extends PyTree implements PyDictCompExpressionTree {

  private final PyExpressionTree keyExpression;
  private final PyToken colon;
  private final PyExpressionTree valueExpression;
  private final PyComprehensionForTree comprehensionFor;

  public PyDictCompExpressionTreeImpl(PyToken openingBrace, PyExpressionTree keyExpression, PyToken colon, PyExpressionTree valueExpression,
                                      PyComprehensionForTree compFor, PyToken closingBrace) {
    super(openingBrace, closingBrace);
    this.keyExpression = keyExpression;
    this.colon = colon;
    this.valueExpression = valueExpression;
    this.comprehensionFor = compFor;
  }

  @Override
  public PyExpressionTree keyExpression() {
    return keyExpression;
  }

  @Override
  public PyToken colonToken() {
    return colon;
  }

  @Override
  public PyExpressionTree valueExpression() {
    return valueExpression;
  }

  @Override
  public PyComprehensionForTree comprehensionFor() {
    return comprehensionFor;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitDictCompExpression(this);
  }

  @Override
  public List<Tree> children() {
    return Arrays.asList(keyExpression, valueExpression, comprehensionFor);
  }

  @Override
  public Kind getKind() {
    return Kind.DICT_COMPREHENSION;
  }
}
