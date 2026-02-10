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

import java.io.IOException;
import org.junit.jupiter.api.Test;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class BuildSystemSourceRootsTest {

  @Test
  void extract_inputFileThrowsIOException_returnsEmptyList() throws IOException {
    InputFile inputFile = mock(InputFile.class);
    when(inputFile.contents()).thenThrow(new IOException("File not readable"));
    assertThat(BuildSystemSourceRoots.extract(inputFile)).isEmpty();
  }

  @Test
  void extract_emptyContent_returnsEmptyList() {
    assertThat(BuildSystemSourceRoots.extract("")).isEmpty();
  }

  @Test
  void extract_invalidToml_returnsEmptyList() {
    assertThat(BuildSystemSourceRoots.extract("[invalid")).isEmpty();
    assertThat(BuildSystemSourceRoots.extract("not toml at all")).isEmpty();
  }

  @Test
  void extract_noToolSection_returnsEmptyList() {
    assertThat(BuildSystemSourceRoots.extract("""
      [project]
      name = "myproject"
      """)).isEmpty();
  }

  @Test
  void extract_emptyToolSection_returnsEmptyList() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool]
      """)).isEmpty();
  }

  // === Setuptools ===

  @Test
  void extract_setuptools_singleWhere() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.setuptools.packages.find]
      where = ["src"]
      """)).containsExactly("src");
  }

  @Test
  void extract_setuptools_multipleWhere() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.setuptools.packages.find]
      where = ["src", "lib"]
      """)).containsExactly("src", "lib");
  }

  @Test
  void extract_setuptools_emptyWhere() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.setuptools.packages.find]
      where = []
      """)).isEmpty();
  }

  @Test
  void extract_setuptools_withOtherOptions() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.setuptools.packages.find]
      where = ["src"]
      include = ["mypackage*"]
      exclude = ["tests*"]
      namespaces = false
      """)).containsExactly("src");
  }

  @Test
  void extract_setuptools_noFindSection() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.setuptools]
      packages = ["mypackage"]
      """)).isEmpty();
  }

  // === Poetry ===

  @Test
  void extract_poetry_singlePackageWithFrom() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.poetry]
      packages = [
        { include = "mypackage", from = "src" }
      ]
      """)).containsExactly("src");
  }

  @Test
  void extract_poetry_multiplePackagesFromDifferentDirs() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.poetry]
      packages = [
        { include = "package1", from = "src" },
        { include = "package2", from = "lib" }
      ]
      """)).containsExactly("src", "lib");
  }

  @Test
  void extract_poetry_multiplePackagesSameDir() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.poetry]
      packages = [
        { include = "package1", from = "src" },
        { include = "package2", from = "src" }
      ]
      """)).containsExactly("src");
  }

  @Test
  void extract_poetry_packageWithoutFrom() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.poetry]
      packages = [
        { include = "mypackage" }
      ]
      """)).isEmpty();
  }

  @Test
  void extract_poetry_emptyPackages() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.poetry]
      packages = []
      """)).isEmpty();
  }

  @Test
  void extract_poetry_onlyDependencies() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.poetry.dependencies]
      python = "^3.9"
      requests = "^2.28"
      """)).isEmpty();
  }

  // === Hatchling ===

  @Test
  void extract_hatchling_sources() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.hatch.build.targets.wheel]
      sources = ["src"]
      """)).containsExactly("src");
  }

  @Test
  void extract_hatchling_multipleSources() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.hatch.build.targets.wheel]
      sources = ["src", "lib"]
      """)).containsExactly("src", "lib");
  }

  @Test
  void extract_hatchling_packagesPath() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.hatch.build.targets.wheel]
      packages = ["src/mypackage"]
      """)).containsExactly("src");
  }

  @Test
  void extract_hatchling_packagesPathWindows(){
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.hatch.build.targets.wheel]
      packages = ["root\\\\mylibrary"]
      """)).containsExactly("root");
  }

  @Test
  void extract_hatchling_multiplePackagesPaths() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.hatch.build.targets.wheel]
      packages = ["src/package1", "lib/package2"]
      """)).containsExactly("src", "lib");
  }

  @Test
  void extract_hatchling_sourcesPreferredOverPackages() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.hatch.build.targets.wheel]
      sources = ["source"]
      packages = ["src/mypackage"]
      """)).containsExactly("source");
  }

  @Test
  void extract_hatchling_packageWithoutSlash() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.hatch.build.targets.wheel]
      packages = ["mypackage"]
      """)).isEmpty();
  }

  @Test
  void extract_hatchling_emptySources() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.hatch.build.targets.wheel]
      sources = []
      """)).isEmpty();
  }

  @Test
  void extract_hatchling_noWheelTarget() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.hatch.build.targets.sdist]
      include = ["src/"]
      """)).isEmpty();
  }

  // === uv_build ===

  @Test
  void extract_uvBuild_moduleRoot() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.uv.build-backend]
      module-root = "src"
      """)).containsExactly("src");
  }

  @Test
  void extract_uvBuild_emptyModuleRoot() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.uv.build-backend]
      module-root = ""
      """)).containsExactly("src");
  }

  @Test
  void extract_uvBuild_noBuildBackend() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.uv]
      dev-dependencies = ["pytest"]
      """)).isEmpty();
  }

  @Test
  void extract_uvBuild_withOtherOptions() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.uv.build-backend]
      module-root = "src"
      module-name = "mymodule"
      """)).containsExactly("src");
  }

  @Test
  void extract_uvBuild_no_root() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.uv.build-backend]
      module-name = "mymodule"
      """)).containsExactly("src");
  }
  // === PDM ===

  @Test
  void extract_pdm_packageDir() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.pdm]
      package-dir = "src"
      """)).containsExactly("src");
  }

  @Test
  void extract_pdm_emptyPackageDir() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.pdm]
      package-dir = ""
      """)).isEmpty();
  }

  @Test
  void extract_pdm_noPackageDir() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.pdm]
      version = { source = "file", path = "src/mypackage/__init__.py" }
      """)).isEmpty();
  }

  // === Flit ===

  @Test
  void extract_flit_withModuleName() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.flit.module]
      name = "mymodule"
      """)).containsExactly("src");
  }

  @Test
  void extract_flit_emptyModuleName() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.flit.module]
      name = ""
      """)).containsExactly("src");
  }

  @Test
  void extract_flit_absentModuleName() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.flit.module]
      other = ""
      """)).isEmpty();
  }

  @Test
  void extract_flit_noModuleSection() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.flit.metadata]
      module = "mymodule"
      """)).isEmpty();
  }

  // === Mixed / Multiple Build Systems ===

  @Test
  void extract_multipleBuildSystems_collectsAll() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.setuptools.packages.find]
      where = ["src"]

      [tool.pdm]
      package-dir = "lib"
      """)).containsExactly("src", "lib");
  }

  @Test
  void extract_multipleBuildSystems_deduplicates() {
    assertThat(BuildSystemSourceRoots.extract("""
      [tool.setuptools.packages.find]
      where = ["src"]

      [tool.pdm]
      package-dir = "src"
      """)).containsExactly("src");
  }

  @Test
  void extract_completePyprojectToml() {
    assertThat(BuildSystemSourceRoots.extract("""
      [build-system]
      requires = ["setuptools>=61.0"]
      build-backend = "setuptools.build_meta"

      [project]
      name = "myproject"
      version = "1.0.0"
      description = "A test project"
      requires-python = ">=3.9"
      dependencies = [
        "requests>=2.28",
        "click>=8.0"
      ]

      [tool.setuptools.packages.find]
      where = ["src"]
      include = ["myproject*"]

      [tool.pytest.ini_options]
      testpaths = ["tests"]
      """)).containsExactly("src");
  }

  // === InputFile API ===

  @Test
  void extract_fromInputFile() {
    var inputFile = TestInputFileBuilder.create("modulekey", "pyproject.toml")
      .setContents("""
        [tool.setuptools.packages.find]
        where = ["src"]
        """)
      .build();

    assertThat(BuildSystemSourceRoots.extract(inputFile)).containsExactly("src");
  }

  @Test
  void extract_fromInputFile_invalidContent() {
    var inputFile = TestInputFileBuilder.create("modulekey", "pyproject.toml")
      .setContents("[invalid")
      .build();

    assertThat(BuildSystemSourceRoots.extract(inputFile)).isEmpty();
  }
}
