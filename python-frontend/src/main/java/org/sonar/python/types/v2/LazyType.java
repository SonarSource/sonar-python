/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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

import java.util.ArrayList;
import java.util.Optional;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.function.Consumer;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeSource;
import org.sonar.python.semantic.v2.LazyTypesContext;

public class LazyType implements PythonType, ResolvableType {

  String importPath;
  private final BlockingQueue<Consumer<PythonType>> consumers;
  private final LazyTypesContext lazyTypesContext;
  private static final String INTERACTION_MESSAGE = "Lazy types should not be interacted with.";

  public LazyType(String importPath, LazyTypesContext lazyTypesContext) {
    this.importPath = importPath;
    this.lazyTypesContext = lazyTypesContext;
    consumers = new LinkedBlockingQueue<>();
  }

  public String importPath() {
    return importPath;
  }

  public LazyType addConsumer(Consumer<PythonType> consumer) {
    consumers.add(consumer);
    return this;
  }

  public LazyType resolve(PythonType type) {
    notifyConsumers(type);
    return this;
  }

  @Override
  public synchronized PythonType resolve() {
    PythonType resolvedType = lazyTypesContext.resolveLazyType(this);
    notifyConsumers(resolvedType);
    return resolvedType;
  }

  private void notifyConsumers(PythonType type) {
    var toNotify = new ArrayList<Consumer<PythonType>>();
    consumers.drainTo(toNotify);
    toNotify.forEach(c -> c.accept(type));
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
  public TriBool isCompatibleWith(PythonType another) {
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
