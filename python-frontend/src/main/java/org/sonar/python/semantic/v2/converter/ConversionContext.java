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
package org.sonar.python.semantic.v2.converter;

import java.util.ArrayDeque;
import java.util.Deque;
import org.sonar.python.index.Descriptor;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TypeOrigin;

public class ConversionContext {
  private final String moduleFqn;
  private final LazyTypesContext lazyTypesContext;
  private final DescriptorToPythonTypeConverter converter;
  private final Deque<PythonType> parents;
  private final TypeOrigin typeOrigin;

  public ConversionContext(String moduleFqn, LazyTypesContext lazyTypesContext, DescriptorToPythonTypeConverter converter, TypeOrigin typeOrigin) {
    this.moduleFqn = moduleFqn;
    this.lazyTypesContext = lazyTypesContext;
    this.converter = converter;
    this.parents = new ArrayDeque<>();
    this.typeOrigin = typeOrigin;
  }

  public LazyTypesContext lazyTypesContext() {
    return lazyTypesContext;
  }

  public TypeOrigin typeOrigin() {
    return typeOrigin;
  }

  public PythonType convert(Descriptor from) {
    return converter.convert(this, from);
  }

  public void pushParent(PythonType pythonType) {
    parents.push(pythonType);
  }

  public PythonType currentParent() {
    return parents.peek();
  }

  public PythonType pollParent() {
    return parents.poll();
  }

  public String moduleFqn() {
    return moduleFqn;
  }
}
