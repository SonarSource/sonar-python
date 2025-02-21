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
package org.sonar.plugins.python.dependency;

import java.util.Set;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.dependency.model.Dependencies;
import org.sonar.plugins.python.dependency.model.Dependency;

import static org.assertj.core.api.Assertions.assertThat;

class DependencyPostProcessorTest {
  @Test
  void test() {
    var dependencies = new Dependencies(Set.of(
      new Dependency("a"),
      new Dependency("b".repeat(99)),
      new Dependency("c".repeat(100)),
      new Dependency("d".repeat(101))));

    var processedDependencies = DependencyPostProcessor.process(dependencies);
    assertThat(processedDependencies.dependencies())
      .map(Dependency::name)
      .containsExactlyInAnyOrder("a", "b".repeat(99));
  }
}
