/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python.dependency;

import java.util.stream.Collectors;
import org.sonar.plugins.python.dependency.model.Dependencies;
import org.sonar.plugins.python.dependency.model.Dependency;

public class DependencyPostProcessor {
  private static final int LENGTH_LIMIT = 100;

  private DependencyPostProcessor() {}

  public static Dependencies process(Dependencies dependencies) {
    var trimmedDependencies = dependencies.dependencies().stream()
      .filter(DependencyPostProcessor::isShortEnough)
      .collect(Collectors.toSet());

    return new Dependencies(trimmedDependencies);
  }

  private static boolean isShortEnough(Dependency dependency) {
    return dependency.name().length() < LENGTH_LIMIT;
  }
}
