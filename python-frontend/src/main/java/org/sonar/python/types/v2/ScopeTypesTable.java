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

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;

public class ScopeTypesTable {
  private final ScopeTypesTable parent;
  private final List<PythonType> types;

  public ScopeTypesTable(ScopeTypesTable parent) {
    this.parent = parent;
    this.types = new ArrayList<>();
  }

  public Optional<PythonType> findType(Predicate<PythonType> predicate) {
    return types.stream()
      .filter(predicate)
      .findFirst()
      .or(() -> Optional.ofNullable(parent)
        .flatMap(p -> p.findType(predicate))
      ).or(Optional::empty);
  }

  public void registerType(PythonType type) {
    types.add(type);
  }
}
