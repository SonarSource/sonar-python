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

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;

public record ModuleType(@Nullable @CheckForNull String name, @Nullable @CheckForNull ModuleType parent, Map<String, PythonType> members) implements PythonType {
  public ModuleType(@Nullable String name) {
    this(name, null);
  }

  public ModuleType(@Nullable String name, @Nullable ModuleType parent) {
    this(name, parent, new HashMap<>());
  }

  @Override
  public boolean isCompatibleWith(PythonType another) {
    return Optional.ofNullable(another)
      .filter(ModuleType.class::isInstance)
      .map(ModuleType.class::cast)
      .map(ModuleType::name)
      .filter(name::equals)
      .isPresent();
  }

  public PythonType resolveMember(String memberName) {
    // FIXME: handle case where type is missing
    return members.getOrDefault(memberName, PythonType.UNKNOWN);
  }

  @Override
  public String toString() {
    return "ModuleType{" +
      "name='" + name + '\'' +
      ", members=" + members +
      '}';
  }
}
