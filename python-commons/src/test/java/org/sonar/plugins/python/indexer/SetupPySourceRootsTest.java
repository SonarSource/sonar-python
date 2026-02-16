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
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.assertj.core.api.Assertions.assertThat;

class SetupPySourceRootsTest {

  @TempDir
  Path tempDir;

  @Test
  void extract_packageDirWithEmptyKey() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup
    setup(package_dir={"": "src"})
    """)).containsExactly("src");
  }

  @Test
  void extract_packageDirWithSeveralValues() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup
    setup(package_dir={"package1": "src1", "package2": "src2", "": "src"})
    """)).containsExactly("src1", "src2", "src");
  }

  @Test
  void extract_packageDirWithIntermediateVariable() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup
    PACKAGE_DIR  = 'package'
    setup(package_dir={"": PACKAGE_DIR})
    """)).containsExactly("package");
  }

  @Test
  void extract_packageDirWithIntermediateVariables() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup
    name = "package"
    package_dir = {name: name}
    setup(package_dir=package_dir)
    """)).containsExactly("package");
  }

  @Test
  void extract_packagesWithFindPackagesAndWhereArgument() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup, find_packages
    setup(package_dir=find_packages(where="src"))
    """)).containsExactly("src");
  }

  @Test
  void extract_packagesAndPackageDir() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup, find_packages
    setup(packages=find_packages(where="src"),
    package_dir={"package1": "src1", "package2": "src2"})
    """)).containsExactly("src", "src1", "src2");
  }

  @Test
  void extract_packagesArgumentWithFindPackages() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup, find_packages
    setup(packages=find_packages(where="src"))
    """)).containsExactly("src");
  }

  @Test
  void extract_packagesWithFindPackagesNoWhere() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup, find_packages
    setup(packages=find_packages())
    """)).isEmpty();
  }

  @Test
  void extract_packageDirWithEmptyStringValue() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup
    setup(package_dir={"": ""})
    """)).isEmpty();
  }

  @Test
  void extract_packageDirWithNonStringValue() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup
    setup(package_dir={"pkg": 123})
    """)).isEmpty();
  }

  @Test
  void extract_noSetupCall() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup
    x = 1
    """)).isEmpty();
  }

  @Test
  void extract_emptyFile() {
    assertThat(SetupPySourceRoots.extract("")).isEmpty();
  }

  @Test
  void extract_malformedPython() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup
    setup(package_dir={"": "src"
    """)).isEmpty();
  }

  @Test
  void extract_setupWithNoRelevantArguments() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup
    setup(name="myproject", version="1.0")
    """)).isEmpty();
  }

  @Test
  void extract_packageDirWithUnresolvedVariable() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup
    setup(package_dir={"": unknown_var})
    """)).isEmpty();
  }

  @Test
  void extract_packageDirWithChainedVariableResolution() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup
    a = "src"
    b = a
    c = b
    setup(package_dir={"": c})
    """)).containsExactly("src");
  }

  @Test
  void extract_findPackagesWithWhereVariable() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup, find_packages
    where_dir = "src"
    setup(packages=find_packages(where=where_dir))
    """)).containsExactly("src");
  }

  @Test
  void extract_multipleSetupCalls() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup
    setup(package_dir={"": "src1"})
    setup(package_dir={"": "src2"})
    """)).containsExactly("src1", "src2");
  }

  @Test
  void extract_nonSetupFunctionCall() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup
    def other_function(package_dir):
        pass
    other_function(package_dir={"": "src"})
    """)).isEmpty();
  }

  @Test
  void extract_packageDirWithVariableAsArgument() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup
    pkg_dir = {"": "src"}
    setup(package_dir=pkg_dir)
    """)).containsExactly("src");
  }

  @Test
  void extract_packagesAsListLiteral() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup
    setup(packages=["mypackage"])
    """)).isEmpty();
  }

  @Test
  void extract_packageDirAsVariable() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup
    dirs = {"": "src", "pkg": "lib"}
    setup(package_dir=dirs)
    """)).containsExactly("src", "lib");
  }

  @Test
  void extract_findPackagesNotASimpleName() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup
    import setuptools
    setup(packages=setuptools.find_packages(where="src"))
    """)).isEmpty();
  }

  @Test
  void extract_duplicateSourceRoots() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup
    setup(
      packages=find_packages(where="src"),
      package_dir={"": "src"}
    )
    """)).containsExactly("src");
  }

  @Test
  void extract_packageDirWithMixedKeyValuePairs() {
    assertThat(SetupPySourceRoots.extract("""
    from setuptools import setup
    setup(package_dir={"": "src", "pkg1": "", "pkg2": None})
    """)).containsExactly("src");
  }

  // === Dictionary unpacking support ===

  @Test
  void extract_setupWithDictUnpacking() {
    assertThat(SetupPySourceRoots.extract("""
      from setuptools import setup
      config = {"package_dir": {"": "src"}}
      setup(**config)
      """)).containsExactly("src");
  }

  @Test
  void extract_setupWithDictUnpackingInline() {
    assertThat(SetupPySourceRoots.extract("""
      from setuptools import setup
      setup(**{"package_dir": {"": "src"}})
      """)).containsExactly("src");
  }

  @Test
  void extract_setupWithDictUnpackingPackages() {
    assertThat(SetupPySourceRoots.extract("""
      from setuptools import setup, find_packages
      config = {"packages": find_packages(where="src")}
      setup(**config)
      """)).containsExactly("src");
  }

  @Test
  void extract_setupWithDictUnpackingBothPackagesAndPackageDir() {
    assertThat(SetupPySourceRoots.extract("""
      from setuptools import setup, find_packages
      config = {
          "packages": find_packages(where="src"),
          "package_dir": {"": "lib"}
      }
      setup(**config)
      """)).containsExactly("src", "lib");
  }

  @Test
  void extract_setupWithDictUnpackingAndRegularArgs() {
    assertThat(SetupPySourceRoots.extract("""
      from setuptools import setup
      base_config = {"package_dir": {"": "src"}}
      setup(**base_config, name="myproject")
      """)).containsExactly("src");
  }

  @Test
  void extract_setupWithDictUnpackingChainedVariable() {
    assertThat(SetupPySourceRoots.extract("""
      from setuptools import setup
      pkg_dir = {"": "src"}
      config = {"package_dir": pkg_dir}
      setup(**config)
      """)).containsExactly("src");
  }

  @Test
  void extract_setupWithDictUnpackingNoRelevantKeys() {
    assertThat(SetupPySourceRoots.extract("""
      from setuptools import setup
      config = {"name": "myproject", "version": "1.0"}
      setup(**config)
      """)).isEmpty();
  }

  @Test
  void extract_setupWithSingleStarUnpacking() {
    // Single star unpacking (*args) should be ignored
    assertThat(SetupPySourceRoots.extract("""
      from setuptools import setup
      args = ["myproject"]
      setup(*args)
      """)).isEmpty();
  }

  @Test
  void extract_setupWithMultipleDictUnpacking() {
    assertThat(SetupPySourceRoots.extract("""
      from setuptools import setup
      config1 = {"package_dir": {"": "src"}}
      config2 = {"package_dir": {"pkg": "lib"}}
      setup(**config1, **config2)
      """)).containsExactly("src", "lib");
  }

  // === extractWithLocation API ===

  @Test
  void extractWithLocation_returnsConfigSourceRoots() throws IOException {
    File file = tempDir.resolve("setup.py").toFile();
    Files.writeString(file.toPath(), """
      from setuptools import setup
      setup(package_dir={"": "src"})
      """);

    ConfigSourceRoots result = SetupPySourceRoots.extractWithLocation(file);

    assertThat(result.configFile()).isEqualTo(file);
    assertThat(result.relativeRoots()).containsExactly("src");
  }

  @Test
  void extractWithLocation_resolvesAbsolutePathsRelativeToConfigFile() throws IOException {
    // Create a subdirectory structure: tempDir/subproject/setup.py
    Path subprojectDir = tempDir.resolve("subproject");
    Files.createDirectories(subprojectDir);
    File file = subprojectDir.resolve("setup.py").toFile();
    Files.writeString(file.toPath(), """
      from setuptools import setup
      setup(package_dir={"": "src"})
      """);

    ConfigSourceRoots result = SetupPySourceRoots.extractWithLocation(file);

    assertThat(result.configFile()).isEqualTo(file);
    assertThat(result.relativeRoots()).containsExactly("src");
    // The absolute path should be relative to the config file's directory, not tempDir
    assertThat(result.toAbsolutePaths()).containsExactly(
      subprojectDir.resolve("src").toFile().getAbsolutePath()
    );
  }

  @Test
  void extractWithLocation_emptyRootsWhenNoConfig() throws IOException {
    File file = tempDir.resolve("setup.py").toFile();
    Files.writeString(file.toPath(), """
      from setuptools import setup
      setup(name="myproject")
      """);

    ConfigSourceRoots result = SetupPySourceRoots.extractWithLocation(file);

    assertThat(result.configFile()).isEqualTo(file);
    assertThat(result.relativeRoots()).isEmpty();
    assertThat(result.toAbsolutePaths()).isEmpty();
  }

  @Test
  void extractWithLocation_multipleRoots() throws IOException {
    Path subprojectDir = tempDir.resolve("app");
    Files.createDirectories(subprojectDir);
    File file = subprojectDir.resolve("setup.py").toFile();
    Files.writeString(file.toPath(), """
      from setuptools import setup
      setup(package_dir={"pkg1": "src", "pkg2": "lib"})
      """);

    ConfigSourceRoots result = SetupPySourceRoots.extractWithLocation(file);

    assertThat(result.toAbsolutePaths()).containsExactly(
      subprojectDir.resolve("src").toFile().getAbsolutePath(),
      subprojectDir.resolve("lib").toFile().getAbsolutePath()
    );
  }

  @Test
  void extractWithLocation_fileNotReadable() {
    File nonExistentFile = new File(tempDir.toFile(), "nonexistent.py");

    ConfigSourceRoots result = SetupPySourceRoots.extractWithLocation(nonExistentFile);

    assertThat(result.relativeRoots()).isEmpty();
  }

  @Test
  void extractWithLocation_malformedContent() throws IOException {
    File file = tempDir.resolve("setup.py").toFile();
    Files.writeString(file.toPath(), "[invalid python");

    ConfigSourceRoots result = SetupPySourceRoots.extractWithLocation(file);

    assertThat(result.relativeRoots()).isEmpty();
  }
}
