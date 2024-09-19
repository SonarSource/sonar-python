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

import java.util.Optional;
import org.sonar.api.Beta;
import org.sonar.plugins.python.api.LocationInFile;

/**
 * PythonType
 */
@Beta
public interface PythonType {
  PythonType UNKNOWN = new UnknownType();

  @Beta
  default String name() {
    return this.toString();
  }

  @Beta
  default Optional<String> displayName() {
    return Optional.empty();
  }

  @Beta
  default Optional<String> instanceDisplayName() {
    return Optional.empty();
  }

  @Beta
  default boolean isCompatibleWith(PythonType another) {
    return true;
  }

  @Beta
  default String key() {
    return name();
  }

  @Beta
  default Optional<PythonType> resolveMember(String memberName) {
    return Optional.empty();
  }

  @Beta
  default TriBool hasMember(String memberName) {
    return TriBool.UNKNOWN;
  }

  @Beta
  default Optional<LocationInFile> definitionLocation() {
    return Optional.empty();
  }

  @Beta
  default PythonType unwrappedType() {
    return this;
  }

  @Beta
  default TypeSource typeSource() {
    return TypeSource.EXACT;
  }

  default PythonType owner() {
    return null;
  }

  @Beta
  default String fullyQualifiedName() {
    return Optional.ofNullable(this.owner())
      .map(owner -> {
        var ownerFQN = owner.fullyQualifiedName();
        if (ownerFQN.isEmpty()) {
          return name();
        }
        return owner.fullyQualifiedName() + "." + name();
      }).orElse(null);
  }
}
