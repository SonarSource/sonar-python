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

import java.util.Comparator;
import java.util.stream.Stream;

public enum TypeSource {
  TYPE_HINT(0),
  EXACT(1);

  private final int score;

  TypeSource(int score) {
    this.score = score;
  }

  public int score() {
    return score;
  }

  public static TypeSource min(TypeSource... typeSources) {
    return Stream.of(typeSources)
      .min(Comparator.comparing(TypeSource::score))
      .orElse(EXACT);
  }
}
