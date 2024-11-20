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

import java.util.Optional;
import org.sonar.api.Beta;
import org.sonar.plugins.python.api.LocationInFile;

/**
 * PythonType
 */
@Beta
public interface PythonType {
  PythonType UNKNOWN = new UnknownType.UnknownTypeImpl();

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
}
