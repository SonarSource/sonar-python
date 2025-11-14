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
package org.sonar.plugins.python.editions;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.plugins.python.editions.RepositoryInfoProvider.RepositoryInfo;

import static org.assertj.core.api.Assertions.assertThat;

class OpenSourceRepositoryInfoProviderTest {
  private static final Path METADATA_DIR = Paths.get("../python-checks/src/main/resources/");

  static Stream<Arguments> repositoryInfo() {
    var provider = new OpenSourceRepositoryInfoProvider();
    return Stream.of(
      Arguments.of(provider.getInfo()),
      Arguments.of(provider.getIPynbInfo())
    );
  }

  @ParameterizedTest
  @MethodSource("repositoryInfo")
  void testPaths(RepositoryInfo repositoryInfo) {
    assertThat(METADATA_DIR.resolve(repositoryInfo.profileLocation())).exists();
  }
}
