/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.types.v2;

import java.util.ArrayDeque;
import java.util.Optional;
import java.util.Queue;
import java.util.function.Consumer;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.python.semantic.v2.LazyTypesContext;

public class LazyType implements PythonType, ResolvableType {

  String importPath;
  private final Queue<Consumer<PythonType>> consumers;
  private final LazyTypesContext lazyTypesContext;
  private static final String INTERACTION_MESSAGE = "Lazy types should not be interacted with.";

  public LazyType(String importPath, LazyTypesContext lazyTypesContext) {
    this.importPath = importPath;
    this.lazyTypesContext = lazyTypesContext;
    consumers = new ArrayDeque<>();
  }

  public String importPath() {
    return importPath;
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
