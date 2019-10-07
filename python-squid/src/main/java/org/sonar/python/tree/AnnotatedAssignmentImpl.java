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

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.AnnotatedAssignment;
import org.sonar.python.api.tree.Expression;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.TreeVisitor;
import org.sonar.python.api.tree.Tree;

public class AnnotatedAssignmentImpl extends PyTree implements AnnotatedAssignment {
  private final Expression variable;
  private final Token colonToken;
  private final Expression annotation;
  private final Token equalToken;
  private final Expression assignedValue;
  private final Token separator;

  public AnnotatedAssignmentImpl(Expression variable, Token colonToken, Expression annotation,
                                 @Nullable Token equalToken, @Nullable Expression assignedValue, @Nullable Token separator) {
    super(variable.firstToken(), assignedValue != null ? assignedValue.lastToken() : annotation.lastToken());
    this.variable = variable;
    this.colonToken = colonToken;
    this.annotation = annotation;
    this.equalToken = equalToken;
    this.assignedValue = assignedValue;
    this.separator = separator;
  }

  @Override
  public Expression variable() {
    return variable;
  }

  @Override
  public Token colonToken() {
    return colonToken;
  }

  @Override
  public Expression annotation() {
    return annotation;
  }

  @CheckForNull
  @Override
  public Token equalToken() {
    return equalToken;
  }

  @CheckForNull
  @Override
  public Expression assignedValue() {
    return assignedValue;
  }

  @CheckForNull
  @Override
  public Token separator() {
    return separator;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitAnnotatedAssignment(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(variable, colonToken, annotation, equalToken, assignedValue, separator).filter(Objects::nonNull).collect(Collectors.toList());
  }

  @Override
  public Kind getKind() {
    return Kind.ANNOTATED_ASSIGNMENT;
  }
}
