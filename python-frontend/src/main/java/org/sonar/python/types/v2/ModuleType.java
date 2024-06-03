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
import java.util.Objects;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.api.Beta;

@Beta
public final class ModuleType implements PythonType {
  @Nullable
  private final String name;
  @Nullable
  private final ModuleType parent;
  private final Map<String, PythonType> members;

  public ModuleType(@Nullable String name, @Nullable ModuleType parent, Map<String, PythonType> members) {
    this.name = name;
    this.parent = parent;
    this.members = members;
  }

  public ModuleType(@Nullable String name) {
    this(name, null);
  }

  public ModuleType(@Nullable String name, @Nullable ModuleType parent) {
    this(name, parent, new HashMap<>());
  }

  @Override
  public Optional<PythonType> resolveMember(String memberName) {
    return Optional.ofNullable(members.get(memberName));
  }

  @Override
  public boolean equals(Object o) {
    // TODO: Find a way how we want to compare modules
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    ModuleType that = (ModuleType) o;
    return Objects.equals(name, that.name) && Objects.equals(members, that.members);
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, members);
  }

  @Override
  public String toString() {
    return "ModuleType{" +
      "name='" + name + '\'' +
      ", members=" + members +
      '}';
  }

  @Override
  @Nullable
  public String name() {
    return name;
  }

  @Nullable
  public ModuleType parent() {
    return parent;
  }

  public Map<String, PythonType> members() {
    return members;
  }

}
