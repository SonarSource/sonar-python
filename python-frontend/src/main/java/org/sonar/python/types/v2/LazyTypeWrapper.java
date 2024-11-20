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
package org.sonar.python.types.v2;

import java.util.Objects;

public class LazyTypeWrapper implements TypeWrapper {
  private PythonType type;

  public LazyTypeWrapper(PythonType type) {
    this.type = type;
    if (type instanceof LazyType lazyType) {
      lazyType.addConsumer(this::resolveLazyType);
    }
  }

  public PythonType type() {
    return TypeUtils.resolved(this.type);
  }

  public void resolveLazyType(PythonType pythonType) {
    if (!(type instanceof LazyType)) {
      throw new IllegalStateException("Trying to resolve an already resolved lazy type.");
    }
    this.type = pythonType;
  }

  public boolean isResolved() {
    return !(type instanceof LazyType);
  }

  public boolean hasImportPath(String importPath) {
    return ((LazyType) type).importPath().equals(importPath);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    LazyTypeWrapper that = (LazyTypeWrapper) o;
    return Objects.equals(type, that.type);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(type);
  }

  @Override
  public String toString() {
    return "LazyTypeWrapper{" +
      "type=" + type +
      '}';
  }
}
