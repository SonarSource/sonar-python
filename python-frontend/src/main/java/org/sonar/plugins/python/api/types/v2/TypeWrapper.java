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
package org.sonar.plugins.python.api.types.v2;

import org.sonar.python.types.v2.LazyTypeWrapper;
import org.sonar.python.types.v2.ResolvableType;
import org.sonar.python.types.v2.SimpleTypeWrapper;

public interface TypeWrapper {

  PythonType type();

  TypeWrapper UNKNOWN_TYPE_WRAPPER = new SimpleTypeWrapper(PythonType.UNKNOWN);

  static TypeWrapper of(PythonType type) {
    if (type instanceof ResolvableType) {
      return new LazyTypeWrapper(type);
    }
    return new SimpleTypeWrapper(type);
  }
}
