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
import java.util.Arrays;
import java.util.List;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyParameterTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.PyTypeAnnotationTree;
import org.sonar.python.api.tree.Tree;

public class PyParameterTreeImpl extends PyTree implements PyParameterTree {

  private final PyNameTree name;
  private final PyTypeAnnotationTree annotation;
  private final PyToken equalToken;
  private final PyExpressionTree defaultValue;
  private final PyToken starToken;

  public PyParameterTreeImpl(AstNode node, @Nullable PyToken starToken, PyNameTree name, @Nullable PyTypeAnnotationTree annotation,
                             @Nullable PyToken equalToken, @Nullable PyExpressionTree defaultValue) {
    super(node);
    this.starToken = starToken;
    this.name = name;
    this.annotation = annotation;
    this.equalToken = equalToken;
    this.defaultValue = defaultValue;
  }

  @CheckForNull
  @Override
  public PyToken starToken() {
    return starToken;
  }

  @Override
  public PyNameTree name() {
    return name;
  }

  @CheckForNull
  @Override
  public PyTypeAnnotationTree typeAnnotation() {
    return annotation;
  }

  @CheckForNull
  @Override
  public PyToken equalToken() {
    return equalToken;
  }

  @CheckForNull
  @Override
  public PyExpressionTree defaultValue() {
    return defaultValue;
  }

  @Override
  public Kind getKind() {
    return Kind.PARAMETER;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitParameter(this);
  }

  @Override
  public List<Tree> children() {
    return Arrays.asList(name, annotation, defaultValue);
  }
}
