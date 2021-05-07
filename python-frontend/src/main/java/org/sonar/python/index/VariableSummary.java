/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.python.index;

import javax.annotation.CheckForNull;
import javax.annotation.Nullable;

class VariableSummary implements Summary {
  private final String name;
  private final String fullyQualifiedName;
  private final String annotatedType;

  public VariableSummary(String name, String fullyQualifiedName, @Nullable String annotatedType) {
    this.name = name;
    this.fullyQualifiedName = fullyQualifiedName;
    this.annotatedType = annotatedType;
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public String fullyQualifiedName() {
    return fullyQualifiedName;
  }

  @CheckForNull
  public String annotatedType() {
    return annotatedType;
  }
}
