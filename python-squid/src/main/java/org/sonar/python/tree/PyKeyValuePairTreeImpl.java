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
import javax.annotation.CheckForNull;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyKeyValuePairTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyKeyValuePairTreeImpl extends PyTree implements PyKeyValuePairTree {

  private final PyToken starStarToken;
  private final PyExpressionTree expression;
  private final PyExpressionTree key;
  private final PyToken colon;
  private final PyExpressionTree value;

  public PyKeyValuePairTreeImpl(PyToken starStarToken, PyExpressionTree expression) {
    super(starStarToken, expression.lastToken());
    this.starStarToken = starStarToken;
    this.expression = expression;
    this.key = null;
    this.colon = null;
    this.value = null;
  }

  public PyKeyValuePairTreeImpl(PyExpressionTree key, PyToken colon, PyExpressionTree value) {
    super(key.firstToken(), value.lastToken());
    this.key = key;
    this.colon = colon;
    this.value = value;
    this.starStarToken = null;
    this.expression = null;
  }

  @CheckForNull
  @Override
  public PyExpressionTree key() {
    return key;
  }

  @CheckForNull
  @Override
  public PyToken colon() {
    return colon;
  }

  @CheckForNull
  @Override
  public PyExpressionTree value() {
    return value;
  }

  @CheckForNull
  @Override
  public PyToken starStarToken() {
    return starStarToken;
  }

  @CheckForNull
  @Override
  public PyExpressionTree expression() {
    return expression;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitKeyValuePair(this);
  }

  @Override
  public List<Tree> children() {
    return Arrays.asList(expression, key, value);
  }

  @Override
  public Kind getKind() {
    return Kind.KEY_VALUE_PAIR;
  }
}
