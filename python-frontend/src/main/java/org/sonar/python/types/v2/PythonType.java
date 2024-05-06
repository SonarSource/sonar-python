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
import org.sonar.plugins.python.api.LocationInFile;

/**
 * PythonType
 */
public interface PythonType {
  PythonType UNKNOWN = new UnknownType();

  default String name() {
    return this.toString();
  }

  default Optional<String> displayName() {
    return Optional.empty();
  }

  default Optional<String> instanceDisplayName() {
    return Optional.empty();
  }

  default boolean isCompatibleWith(PythonType another) {
    return true;
  }

  default String key() {
    return name();
  }

  default Optional<PythonType> resolveMember(String memberName) {
    return Optional.empty();
  }

  default TriBool hasMember(String memberName) {
    return TriBool.UNKNOWN;
  }

  default Optional<LocationInFile> definitionLocation() {
    return Optional.empty();
  }

  default PythonType unwrappedType() {
    return this;
  }
}
