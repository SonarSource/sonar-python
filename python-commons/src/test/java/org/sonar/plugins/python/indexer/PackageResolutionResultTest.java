/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.PYPROJECT_TOML);
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.SETUPTOOLS);
  }

  @Test
  void fromSetupPy_creates_correct_result() {
    List<String> roots = List.of("/project/src");
    var result = PackageResolutionResult.fromSetupPy(roots);

    assertThat(result.roots()).containsExactly("/project/src");
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.SETUP_PY);
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.NONE);
  }

  @Test
  void fromSonarSources_creates_correct_result() {
    List<String> roots = List.of("/project/src", "/project/lib");
    var result = PackageResolutionResult.fromSonarSources(roots);

    assertThat(result.roots()).containsExactly("/project/src", "/project/lib");
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.SONAR_SOURCES);
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.NONE);
  }

  @Test
  void fromConventionalFolders_creates_correct_result() {
    List<String> roots = List.of("/project/src");
    var result = PackageResolutionResult.fromConventionalFolders(roots);

    assertThat(result.roots()).containsExactly("/project/src");
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.CONVENTIONAL_FOLDERS);
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.NONE);
  }

  @Test
  void fromBaseDir_creates_correct_result() {
    List<String> roots = List.of("/project");
    var result = PackageResolutionResult.fromBaseDir(roots);

    assertThat(result.roots()).containsExactly("/project");
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.BASE_DIR);
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.NONE);
  }

  @Test
  void fromLegacyInitPy_creates_correct_result() {
    List<String> roots = List.of("/project", "/project/other");
    var result = PackageResolutionResult.fromLegacyInitPy(roots);

    assertThat(result.roots()).containsExactly("/project", "/project/other");
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.LEGACY_INIT_PY);
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
    assertThat(PackageResolutionResult.PrimaryResolutionMethod.values()).containsExactly(
      PackageResolutionResult.PrimaryResolutionMethod.PYPROJECT_TOML,
      PackageResolutionResult.PrimaryResolutionMethod.SETUP_PY,
      PackageResolutionResult.PrimaryResolutionMethod.PYPROJECT_AND_SETUP_PY,
      PackageResolutionResult.PrimaryResolutionMethod.SONAR_SOURCES,
      PackageResolutionResult.PrimaryResolutionMethod.CONVENTIONAL_FOLDERS,
      PackageResolutionResult.PrimaryResolutionMethod.BASE_DIR,
      PackageResolutionResult.PrimaryResolutionMethod.LEGACY_INIT_PY
    );
  }

  @Test
  void fromBothPyProjectAndSetupPy_creates_correct_result() {
    List<String> roots = List.of("/project/src", "/project/lib");
    var result = PackageResolutionResult.fromBothPyProjectAndSetupPy(roots, PackageResolutionResult.BuildSystem.SETUPTOOLS);

    assertThat(result.roots()).containsExactly("/project/src", "/project/lib");
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.PYPROJECT_AND_SETUP_PY);
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.SETUPTOOLS);
  }
}
