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

import java.util.Arrays;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;

public class TypeChecker {

  private final ProjectLevelTypeTable projectLevelTypeTable;

  public TypeChecker(ProjectLevelTypeTable projectLevelTypeTable) {
    this.projectLevelTypeTable = projectLevelTypeTable;
  }

  public TypeCheckBuilder typeCheckBuilder() {
    return new TypeCheckBuilder(projectLevelTypeTable);
  }

  public TypeCheckBuilder and(TypeCheckBuilder... builders) {
    TypeCheckBuilder result = Arrays.stream(builders)
      .reduce(TypeCheckBuilder::and)
      .orElseThrow(() -> new IllegalArgumentException("At least one builder is required"));
    return new TypeCheckBuilder(projectLevelTypeTable, result.predicate);
  }

  public TypeCheckBuilder or(TypeCheckBuilder... builders) {
    TypeCheckBuilder result = Arrays.stream(builders)
      .reduce(TypeCheckBuilder::or)
      .orElseThrow(() -> new IllegalArgumentException("At least one builder is required"));
    return new TypeCheckBuilder(projectLevelTypeTable, result.predicate);
  }
}
