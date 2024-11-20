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

import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.tree.TypeAnnotation;

public class ParameterImpl extends PyTree implements Parameter {

  private final Name name;
  private final TypeAnnotation annotation;
  private final Token equalToken;
  private final Expression defaultValue;
  private final Token starToken;

  public ParameterImpl(@Nullable Token starToken, Name name, @Nullable TypeAnnotation annotation,
                       @Nullable Token equalToken, @Nullable Expression defaultValue) {
    this.starToken = starToken;
    this.name = name;
    this.annotation = annotation;
    this.equalToken = equalToken;
    this.defaultValue = defaultValue;
  }

  /**
   * constructor for star parameter syntax.
   * def fun(arg1, *, arg2)
   */
  public ParameterImpl(Token starToken) {
    this.starToken = starToken;
    this.name = null;
    this.annotation = null;
    this.equalToken = null;
    this.defaultValue = null;
  }

  @CheckForNull
  @Override
  public Token starToken() {
    return starToken;
  }

  @CheckForNull
  @Override
  public Name name() {
    return name;
  }

  @CheckForNull
  @Override
  public TypeAnnotation typeAnnotation() {
    return annotation;
  }

  @CheckForNull
  @Override
  public Token equalToken() {
    return equalToken;
  }

  @CheckForNull
  @Override
  public Expression defaultValue() {
    return defaultValue;
  }

  @Override
  public Kind getKind() {
    return Kind.PARAMETER;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitParameter(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(starToken, name, annotation, equalToken, defaultValue).filter(Objects::nonNull).toList();
  }
}
