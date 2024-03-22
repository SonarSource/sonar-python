/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.tree.TypeParam;

public class TypeParamImpl extends PyTree implements TypeParam {

  private final Token starToken;
  private final Name name;
  private final TypeAnnotation annotation;

  public TypeParamImpl(@Nullable Token starToken, Name name, @Nullable TypeAnnotation annotation) {
    this.starToken = starToken;
    this.name = name;
    this.annotation = annotation;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitTypeParam(this);
  }

  @Override
  public Kind getKind() {
    return Kind.TYPE_PARAM;
  }

  @CheckForNull
  @Override
  public Token starToken() {
    return starToken;
  }

  @Override
  public Name name() {
    return name;
  }

  @CheckForNull
  @Override
  public TypeAnnotation typeAnnotation() {
    return annotation;
  }

  @Override
  List<Tree> computeChildren() {
    return Stream.of(starToken, name, annotation).filter(Objects::nonNull).toList();
  }
}
