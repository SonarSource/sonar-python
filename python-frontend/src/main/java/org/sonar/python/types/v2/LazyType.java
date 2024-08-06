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
package org.sonar.python.types.v2;

import java.util.ArrayDeque;
import java.util.Optional;
import java.util.Queue;
import java.util.function.Consumer;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.python.semantic.v2.LazyTypesContext;

public class LazyType implements PythonType {

  String fullyQualifiedName;
  private final Queue<Consumer<PythonType>> consumers;
  private final LazyTypesContext lazyTypesContext;
  private static final String INTERACTION_MESSAGE = "Lazy types should not be interacted with.";

  public LazyType(String fullyQualifiedName, LazyTypesContext lazyTypesContext) {
    this.fullyQualifiedName = fullyQualifiedName;
    this.lazyTypesContext = lazyTypesContext;
    consumers = new ArrayDeque<>();
  }

  public String fullyQualifiedName() {
    return fullyQualifiedName;
  }

  public LazyType addConsumer(Consumer<PythonType> consumer) {
    consumers.add(consumer);
    return this;
  }

  public LazyType resolve(PythonType type) {
    consumers.forEach(c -> c.accept(type));
    consumers.clear();
    return this;
  }

  public PythonType resolve() {
    PythonType resolvedType = lazyTypesContext.resolveLazyType(this);
    consumers.forEach(c -> c.accept(resolvedType));
    consumers.clear();
    return resolvedType;
  }

  @Override
  public PythonType unwrappedType() {
    throw new IllegalStateException(INTERACTION_MESSAGE);
  }

  @Override
  public String name() {
    throw new IllegalStateException(INTERACTION_MESSAGE);
  }

  @Override
  public Optional<String> displayName() {
    throw new IllegalStateException(INTERACTION_MESSAGE);
  }

  @Override
  public Optional<String> instanceDisplayName() {
    throw new IllegalStateException(INTERACTION_MESSAGE);
  }

  @Override
  public boolean isCompatibleWith(PythonType another) {
    throw new IllegalStateException(INTERACTION_MESSAGE);
  }

  @Override
  public String key() {
    throw new IllegalStateException(INTERACTION_MESSAGE);
  }

  @Override
  public Optional<PythonType> resolveMember(String memberName) {
    throw new IllegalStateException(INTERACTION_MESSAGE);
  }

  @Override
  public TriBool hasMember(String memberName) {
    throw new IllegalStateException(INTERACTION_MESSAGE);
  }

  @Override
  public Optional<LocationInFile> definitionLocation() {
    throw new IllegalStateException(INTERACTION_MESSAGE);
  }

  @Override
  public TypeSource typeSource() {
    throw new IllegalStateException(INTERACTION_MESSAGE);
  }
}
