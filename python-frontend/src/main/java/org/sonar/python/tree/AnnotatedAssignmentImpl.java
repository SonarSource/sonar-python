/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.tree;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;

public class AnnotatedAssignmentImpl extends SimpleStatement implements AnnotatedAssignment {
  private final Expression variable;
  private final Token equalToken;
  private final Expression assignedValue;
  private final Separators separators;
  private final TypeAnnotation annotation;

  public AnnotatedAssignmentImpl(Expression variable, TypeAnnotation annotation, @Nullable Token equalToken,
                                 @Nullable Expression assignedValue, Separators separators) {
    this.variable = variable;
    this.annotation = annotation;
    this.equalToken = equalToken;
    this.assignedValue = assignedValue;
    this.separators = separators;
  }

  @Override
  public Expression variable() {
    return variable;
  }

  @Override
  public TypeAnnotation annotation() {
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
    return separators.last();
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitAnnotatedAssignment(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(Arrays.asList(variable, annotation, equalToken, assignedValue), separators.elements())
      .flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }

  @Override
  public Kind getKind() {
    return Kind.ANNOTATED_ASSIGNMENT;
  }
}
