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

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.assertj.core.api.Assertions.assertThat;
class PyProjectTomlSourceRootsTest {

  @TempDir
  Path tempDir;

  // === Helper functions to simulate the toml file ===

  public List<String> extract(String content) {
    File file = tempDir.resolve("pyproject.toml").toFile();
    try {
      Files.writeString(file.toPath(), content);
      return PyProjectTomlSourceRoots.extract(file);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public PyProjectExtractionResult extractWithBuildSystem(String content) {
    File file = tempDir.resolve("pyproject.toml").toFile();
    try {
      Files.writeString(file.toPath(), content);
      return PyProjectTomlSourceRoots.extractWithBuildSystem(file);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  // === Error handling ===

  @Test
  void extractWithLocation_fileNotReadable_returnsEmptyRoots() {
    File nonExistentFile = new File(tempDir.toFile(), "nonexistent.toml");
    ConfigSourceRoots result = PyProjectTomlSourceRoots.extractWithLocation(nonExistentFile);
    assertThat(result.relativeRoots()).isEmpty();
  }

  @Test
  void extract_emptyContent_returnsEmptyList() {
    assertThat(extract("")).isEmpty();
  }

  @Test
  void extract_invalidToml_returnsEmptyList() {
    assertThat(extract("[invalid")).isEmpty();
    assertThat(extract("not toml at all")).isEmpty();
  }

  @Test
  void extract_noToolSection_returnsEmptyList() {
    assertThat(extract("""
      [project]
      name = "myproject"
      """)).isEmpty();
  }

  @Test
  void extract_emptyToolSection_returnsEmptyList() {
    assertThat(extract("""
      [tool]
      """)).isEmpty();
  }

  // === Setuptools ===

  @Test
  void extract_setuptools_singleWhere() {
    assertThat(extract("""
      [tool.setuptools.packages.find]
      where = ["src"]
      """)).containsExactly("src");
  }

  @Test
  void extract_setuptools_multipleWhere() {
    assertThat(extract("""
      [tool.setuptools.packages.find]
      where = ["src", "lib"]
      """)).containsExactly("src", "lib");
  }

  @Test
  void extract_setuptools_emptyWhere() {
    assertThat(extract("""
      [tool.setuptools.packages.find]
      where = []
      """)).isEmpty();
  }

  @Test
  void extract_setuptools_withOtherOptions() {
    assertThat(extract("""
      [tool.setuptools.packages.find]
      where = ["src"]
      include = ["mypackage*"]
      exclude = ["tests*"]
      namespaces = false
      """)).containsExactly("src");
  }

  @Test
  void extract_setuptools_noFindSection() {
    assertThat(extract("""
      [tool.setuptools]
      packages = ["mypackage"]
      """)).isEmpty();
  }

  // === Poetry ===

  @Test
  void extract_poetry_singlePackageWithFrom() {
    assertThat(extract("""
      [tool.poetry]
      packages = [
        { include = "mypackage", from = "src" }
      ]
      """)).containsExactly("src");
  }

  @Test
  void extract_poetry_multiplePackagesFromDifferentDirs() {
    assertThat(extract("""
      [tool.poetry]
      packages = [
        { include = "package1", from = "src" },
        { include = "package2", from = "lib" }
      ]
      """)).containsExactly("src", "lib");
  }

  @Test
  void extract_poetry_multiplePackagesSameDir() {
    assertThat(extract("""
      [tool.poetry]
      packages = [
        { include = "package1", from = "src" },
        { include = "package2", from = "src" }
      ]
      """)).containsExactly("src");
  }

  @Test
  void extract_poetry_packageWithoutFrom() {
    assertThat(extract("""
      [tool.poetry]
      packages = [
        { include = "mypackage" }
      ]
      """)).isEmpty();
  }

  @Test
  void extract_poetry_emptyPackages() {
    assertThat(extract("""
      [tool.poetry]
      packages = []
      """)).isEmpty();
  }

  @Test
  void extract_poetry_onlyDependencies() {
    assertThat(extract("""
      [tool.poetry.dependencies]
      python = "^3.9"
      requests = "^2.28"
      """)).isEmpty();
  }

  // === Hatchling ===

  @Test
  void extract_hatchling_sources() {
    assertThat(extract("""
      [tool.hatch.build.targets.wheel]
      sources = ["src"]
      """)).containsExactly("src");
  }

  @Test
  void extract_hatchling_multipleSources() {
    assertThat(extract("""
      [tool.hatch.build.targets.wheel]
      sources = ["src", "lib"]
      """)).containsExactly("src", "lib");
  }

  @Test
  void extract_hatchling_packagesPath() {
    assertThat(extract("""
      [tool.hatch.build.targets.wheel]
      packages = ["src/mypackage"]
      """)).containsExactly("src");
  }

  @Test
  void extract_hatchling_packagesPathWindows(){
    assertThat(extract("""
      [tool.hatch.build.targets.wheel]
      packages = ["root\\\\mylibrary"]
      """)).containsExactly("root");
  }

  @Test
  void extract_hatchling_multiplePackagesPaths() {
    assertThat(extract("""
      [tool.hatch.build.targets.wheel]
      packages = ["src/package1", "lib/package2"]
      """)).containsExactly("src", "lib");
  }

  @Test
  void extract_hatchling_sourcesPreferredOverPackages() {
    assertThat(extract("""
      [tool.hatch.build.targets.wheel]
      sources = ["source"]
      packages = ["src/mypackage"]
      """)).containsExactly("source");
  }

  @Test
  void extract_hatchling_packageWithoutSlash() {
    assertThat(extract("""
      [tool.hatch.build.targets.wheel]
      packages = ["mypackage"]
      """)).isEmpty();
  }

  @Test
  void extract_hatchling_emptySources() {
    assertThat(extract("""
      [tool.hatch.build.targets.wheel]
      sources = []
      """)).isEmpty();
  }

  @Test
  void extract_hatchling_noWheelTarget() {
    assertThat(extract("""
      [tool.hatch.build.targets.sdist]
      include = ["src/"]
      """)).isEmpty();
  }

  // === uv_build ===

  @Test
  void extract_uvBuild_moduleRoot() {
    assertThat(extract("""
      [tool.uv.build-backend]
      module-root = "src"
      """)).containsExactly("src");
  }

  @Test
  void extract_uvBuild_emptyModuleRoot() {
    assertThat(extract("""
      [tool.uv.build-backend]
      module-root = ""
      """)).containsExactly("src");
  }

  @Test
  void extract_uvBuild_noBuildBackend() {
    assertThat(extract("""
      [tool.uv]
      dev-dependencies = ["pytest"]
      """)).isEmpty();
  }

  @Test
  void extract_uvBuild_withOtherOptions() {
    assertThat(extract("""
      [tool.uv.build-backend]
      module-root = "src"
      module-name = "mymodule"
      """)).containsExactly("src");
  }

  @Test
  void extract_uvBuild_no_root() {
    assertThat(extract("""
      [tool.uv.build-backend]
      module-name = "mymodule"
      """)).containsExactly("src");
  }

  @Test
  void extract_uvBuild_fromBuildSystem() {
    assertThat(extract("""
      [build-system]
      build-backend = "uv_build"
      """)).containsExactly("src");
  }

  @Test
  void extract_uvBuild_fromBuildSystemWithRequires() {
    assertThat(extract("""
      [build-system]
      requires = ["uv>=0.5.15"]
      build-backend = "uv_build"
      """)).containsExactly("src");
  }

  @Test
  void extract_otherBuildBackend_returnsEmpty() {
    assertThat(extract("""
      [build-system]
      build-backend = "setuptools.build_meta"
      """)).isEmpty();
  }

  // === PDM ===

  @Test
  void extract_pdm_packageDir() {
    assertThat(extract("""
      [tool.pdm]
      package-dir = "src"
      """)).containsExactly("src");
  }

  @Test
  void extract_pdm_emptyPackageDir() {
    assertThat(extract("""
      [tool.pdm]
      package-dir = ""
      """)).isEmpty();
  }

  @Test
  void extract_pdm_noPackageDir() {
    assertThat(extract("""
      [tool.pdm]
      version = { source = "file", path = "src/mypackage/__init__.py" }
      """)).isEmpty();
  }

  // === Flit ===

  @Test
  void extract_flit_withModuleName() {
    assertThat(extract("""
      [tool.flit.module]
      name = "mymodule"
      """)).containsExactly("src");
  }

  @Test
  void extract_flit_emptyModuleName() {
    assertThat(extract("""
      [tool.flit.module]
      name = ""
      """)).containsExactly("src");
  }

  @Test
  void extract_flit_absentModuleName() {
    assertThat(extract("""
      [tool.flit.module]
      other = ""
      """)).isEmpty();
  }

  @Test
  void extract_flit_noModuleSection() {
    assertThat(extract("""
      [tool.flit.metadata]
      module = "mymodule"
      """)).isEmpty();
  }

  // === Mixed / Multiple Build Systems ===

  @Test
  void extract_multipleBuildSystems_collectsAll() {
    assertThat(extract("""
      [tool.setuptools.packages.find]
      where = ["src"]

      [tool.pdm]
      package-dir = "lib"
      """)).containsExactly("src", "lib");
  }

  @Test
  void extract_multipleBuildSystems_deduplicates() {
    assertThat(extract("""
      [tool.setuptools.packages.find]
      where = ["src"]

      [tool.pdm]
      package-dir = "src"
      """)).containsExactly("src");
  }

  @Test
  void extract_completePyprojectToml() {
    assertThat(extract("""
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

  // === extractWithLocation API ===

  @Test
  void extractWithLocation_returnsConfigSourceRoots() throws IOException {
    File file = tempDir.resolve("pyproject.toml").toFile();
    Files.writeString(file.toPath(), """
        [tool.setuptools.packages.find]
        where = ["src"]
        """);

    ConfigSourceRoots result = PyProjectTomlSourceRoots.extractWithLocation(file);

    assertThat(result.configFile()).isEqualTo(file);
    assertThat(result.relativeRoots()).containsExactly("src");
  }

  @Test
  void extractWithLocation_resolvesAbsolutePathsRelativeToConfigFile() throws IOException {
    // Create a subdirectory structure: tempDir/subproject/pyproject.toml
    Path subprojectDir = tempDir.resolve("subproject");
    Files.createDirectories(subprojectDir);
    File file = subprojectDir.resolve("pyproject.toml").toFile();
    Files.writeString(file.toPath(), """
        [build-system]
        build-backend = "uv_build"
        """);

    ConfigSourceRoots result = PyProjectTomlSourceRoots.extractWithLocation(file);

    assertThat(result.configFile()).isEqualTo(file);
    assertThat(result.relativeRoots()).containsExactly("src");
    // The absolute path should be relative to the config file's directory, not tempDir
    assertThat(result.toAbsolutePaths()).containsExactly(
      subprojectDir.resolve("src").toFile().getAbsolutePath()
    );
  }

  @Test
  void extractWithLocation_emptyRootsWhenNoConfig() throws IOException {
    File file = tempDir.resolve("pyproject.toml").toFile();
    Files.writeString(file.toPath(), """
        [project]
        name = "myproject"
        """);

    ConfigSourceRoots result = PyProjectTomlSourceRoots.extractWithLocation(file);

    assertThat(result.configFile()).isEqualTo(file);
    assertThat(result.relativeRoots()).isEmpty();
    assertThat(result.toAbsolutePaths()).isEmpty();
  }

  @Test
  void extractWithLocation_multipleRoots() throws IOException {
    Path subprojectDir = tempDir.resolve("app");
    Files.createDirectories(subprojectDir);
    File file = subprojectDir.resolve("pyproject.toml").toFile();
    Files.writeString(file.toPath(), """
        [tool.setuptools.packages.find]
        where = ["src", "lib"]
        """);

    ConfigSourceRoots result = PyProjectTomlSourceRoots.extractWithLocation(file);

    assertThat(result.toAbsolutePaths()).containsExactly(
      subprojectDir.resolve("src").toFile().getAbsolutePath(),
      subprojectDir.resolve("lib").toFile().getAbsolutePath()
    );
  }

  @Test
  void extractWithLocation_invalidContent() throws IOException {
    File file = tempDir.resolve("pyproject.toml").toFile();
    Files.writeString(file.toPath(), "[invalid");

    ConfigSourceRoots result = PyProjectTomlSourceRoots.extractWithLocation(file);

    assertThat(result.relativeRoots()).isEmpty();
  }

  // === Build System Detection Tests ===

  @Test
  void extractWithBuildSystem_setuptools_detectsBuildSystem() {
    var result = extractWithBuildSystem("""
      [tool.setuptools.packages.find]
      where = ["src"]
      """);

    assertThat(result.relativeRoots()).containsExactly("src");
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.SETUPTOOLS);
  }

  @Test
  void extractWithBuildSystem_poetry_detectsBuildSystem() {
    var result = extractWithBuildSystem("""
      [tool.poetry]
      packages = [{ include = "mypackage", from = "src" }]
      """);

    assertThat(result.relativeRoots()).containsExactly("src");
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.POETRY);
  }

  @Test
  void extractWithBuildSystem_hatchling_detectsBuildSystem() {
    var result = extractWithBuildSystem("""
      [tool.hatch.build.targets.wheel]
      sources = ["src"]
      """);

    assertThat(result.relativeRoots()).containsExactly("src");
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.HATCHLING);
  }

  @Test
  void extractWithBuildSystem_uvBuild_detectsBuildSystem() {
    var result = extractWithBuildSystem("""
      [tool.uv.build-backend]
      module-root = "src"
      """);

    assertThat(result.relativeRoots()).containsExactly("src");
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.UV_BUILD);
  }

  @Test
  void extractWithBuildSystem_pdm_detectsBuildSystem() {
    var result = extractWithBuildSystem("""
      [tool.pdm]
      package-dir = "src"
      """);

    assertThat(result.relativeRoots()).containsExactly("src");
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.PDM);
  }

  @Test
  void extractWithBuildSystem_flit_detectsBuildSystem() {
    var result = extractWithBuildSystem("""
      [tool.flit.module]
      name = "mymodule"
      """);

    assertThat(result.relativeRoots()).containsExactly("src");
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.FLIT);
  }

  @Test
  void extractWithBuildSystem_multipleBuildSystems_detectsMultiple() {
    var result = extractWithBuildSystem("""
      [tool.setuptools.packages.find]
      where = ["src"]

      [tool.pdm]
      package-dir = "lib"
      """);

    assertThat(result.relativeRoots()).containsExactly("src", "lib");
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.MULTIPLE);
  }

  @Test
  void extractWithBuildSystem_emptyContent_returnsNone() {
    var result = extractWithBuildSystem("");

    assertThat(result.relativeRoots()).isEmpty();
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.NONE);
  }

  @Test
  void extractWithBuildSystem_noToolSection_returnsNone() {
    var result = extractWithBuildSystem("""
      [project]
      name = "myproject"
      """);

    assertThat(result.relativeRoots()).isEmpty();
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.NONE);
  }

  @Test
  void extractWithBuildSystem_fromFile() throws IOException {
    File file = tempDir.resolve("pyproject.toml").toFile();
    Files.writeString(file.toPath(), """
        [tool.setuptools.packages.find]
        where = ["src"]
        """);

    var result = PyProjectTomlSourceRoots.extractWithBuildSystem(file);

    assertThat(result.relativeRoots()).containsExactly("src");
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.SETUPTOOLS);
  }

  @Test
  void extractWithBuildSystem_fromFileThrowsIOException(){
    File file = tempDir.resolve("pyproject.toml").toFile();
    var result = PyProjectTomlSourceRoots.extractWithBuildSystem(file);
    assertThat(result.relativeRoots()).isEmpty();
    assertThat(result.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.NONE);
  }
}
