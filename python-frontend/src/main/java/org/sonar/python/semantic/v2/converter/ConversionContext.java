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
package org.sonar.python.semantic.v2.converter;

import java.util.ArrayDeque;
import java.util.Deque;
import org.sonar.python.index.Descriptor;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TypeOrigin;

public class ConversionContext {
  private final LazyTypesContext lazyTypesContext;
  private final DescriptorToPythonTypeConverter converter;
  private final Deque<PythonType> parents;
  private final TypeOrigin typeOrigin;

  public ConversionContext(LazyTypesContext lazyTypesContext, DescriptorToPythonTypeConverter converter, TypeOrigin typeOrigin) {
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
}
