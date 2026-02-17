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
package org.sonar.plugins.python.indexer;

import java.util.List;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class PackageResolutionResultTest {

  @Test
  void fromPyProjectToml_creates_correct_result() {
    List<String> roots = List.of("/project/src");
    var result = PackageResolutionResult.fromPyProjectToml(roots, PackageResolutionResult.BuildSystem.SETUPTOOLS);

    assertThat(result.roots()).containsExactly("/project/src");
    assertThat(result.method()).isEqualTo(PackageResolutionResult.ResolutionMethod.PYPROJECT_TOML);
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.SETUPTOOLS);
  }

  @Test
  void fromSetupPy_creates_correct_result() {
    List<String> roots = List.of("/project/src");
    var result = PackageResolutionResult.fromSetupPy(roots);

    assertThat(result.roots()).containsExactly("/project/src");
    assertThat(result.method()).isEqualTo(PackageResolutionResult.ResolutionMethod.SETUP_PY);
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.NONE);
  }

  @Test
  void fromSonarSources_creates_correct_result() {
    List<String> roots = List.of("/project/src", "/project/lib");
    var result = PackageResolutionResult.fromSonarSources(roots);

    assertThat(result.roots()).containsExactly("/project/src", "/project/lib");
    assertThat(result.method()).isEqualTo(PackageResolutionResult.ResolutionMethod.SONAR_SOURCES);
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.NONE);
  }

  @Test
  void fromConventionalFolders_creates_correct_result() {
    List<String> roots = List.of("/project/src");
    var result = PackageResolutionResult.fromConventionalFolders(roots);

    assertThat(result.roots()).containsExactly("/project/src");
    assertThat(result.method()).isEqualTo(PackageResolutionResult.ResolutionMethod.CONVENTIONAL_FOLDERS);
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.NONE);
  }

  @Test
  void fromBaseDir_creates_correct_result() {
    List<String> roots = List.of("/project");
    var result = PackageResolutionResult.fromBaseDir(roots);

    assertThat(result.roots()).containsExactly("/project");
    assertThat(result.method()).isEqualTo(PackageResolutionResult.ResolutionMethod.BASE_DIR);
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.NONE);
  }

  @Test
  void all_build_systems_exist() {
    assertThat(PackageResolutionResult.BuildSystem.values()).containsExactly(
      PackageResolutionResult.BuildSystem.SETUPTOOLS,
      PackageResolutionResult.BuildSystem.POETRY,
      PackageResolutionResult.BuildSystem.HATCHLING,
      PackageResolutionResult.BuildSystem.UV_BUILD,
      PackageResolutionResult.BuildSystem.UV_BUILD_DEFAULT_MODULE,
      PackageResolutionResult.BuildSystem.PDM,
      PackageResolutionResult.BuildSystem.FLIT,
      PackageResolutionResult.BuildSystem.MULTIPLE,
      PackageResolutionResult.BuildSystem.NONE
    );
  }

  @Test
  void all_resolution_methods_exist() {
    assertThat(PackageResolutionResult.ResolutionMethod.values()).containsExactly(
      PackageResolutionResult.ResolutionMethod.PYPROJECT_TOML,
      PackageResolutionResult.ResolutionMethod.SETUP_PY,
      PackageResolutionResult.ResolutionMethod.PYPROJECT_AND_SETUP_PY,
      PackageResolutionResult.ResolutionMethod.SONAR_SOURCES,
      PackageResolutionResult.ResolutionMethod.CONVENTIONAL_FOLDERS,
      PackageResolutionResult.ResolutionMethod.BASE_DIR
    );
  }

  @Test
  void fromBothPyProjectAndSetupPy_creates_correct_result() {
    List<String> roots = List.of("/project/src", "/project/lib");
    var result = PackageResolutionResult.fromBothPyProjectAndSetupPy(roots, PackageResolutionResult.BuildSystem.SETUPTOOLS);

    assertThat(result.roots()).containsExactly("/project/src", "/project/lib");
    assertThat(result.method()).isEqualTo(PackageResolutionResult.ResolutionMethod.PYPROJECT_AND_SETUP_PY);
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.SETUPTOOLS);
  }
}
