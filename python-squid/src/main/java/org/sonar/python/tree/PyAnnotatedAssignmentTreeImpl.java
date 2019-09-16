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

import com.sonar.sslr.api.Token;
import java.util.Arrays;
import java.util.List;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.PyAnnotatedAssignmentTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyAnnotatedAssignmentTreeImpl extends PyTree implements PyAnnotatedAssignmentTree {
  private final PyExpressionTree variable;
  private final Token colonToken;
  private final PyExpressionTree annotation;
  private final Token equalToken;
  private final PyExpressionTree assignedValue;

  public PyAnnotatedAssignmentTreeImpl(PyExpressionTree variable, Token colonToken, PyExpressionTree annotation,
                                       @Nullable Token equalToken, @Nullable PyExpressionTree assignedValue) {
    super(variable.firstToken(), assignedValue != null ? assignedValue.lastToken() : annotation.lastToken());
    this.variable = variable;
    this.colonToken = colonToken;
    this.annotation = annotation;
    this.equalToken = equalToken;
    this.assignedValue = assignedValue;
  }

  @Override
  public PyExpressionTree variable() {
    return variable;
  }

  @Override
  public Token colonToken() {
    return colonToken;
  }

  @Override
  public PyExpressionTree annotation() {
    return annotation;
  }

  @CheckForNull
  @Override
  public Token equalToken() {
    return equalToken;
  }

  @CheckForNull
  @Override
  public PyExpressionTree assignedValue() {
    return assignedValue;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitAnnotatedAssignment(this);
  }

  @Override
  public List<Tree> children() {
    return Arrays.asList(variable, annotation, assignedValue);
  }

  @Override
  public Kind getKind() {
    return Kind.ANNOTATED_ASSIGNMENT;
  }
}
