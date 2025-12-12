/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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

import java.util.Objects;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.types.v2.PythonType;

public class SpecialFormType implements PythonType {

  private final String fullyQualifiedName;

  public SpecialFormType(String fullyQualifiedName) {
    this.fullyQualifiedName = fullyQualifiedName;
  }

  public String fullyQualifiedName() {
    return fullyQualifiedName;
  }

  @Override
  public int hashCode() {
    return Objects.hash(fullyQualifiedName);
  }

  @Override
  public boolean equals(@Nullable Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof SpecialFormType)) {
      return false;
    }
    SpecialFormType other = (SpecialFormType) obj;
    return Objects.equals(fullyQualifiedName, other.fullyQualifiedName);
  }
}
