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

class NamespacePackageAnalyzerTest {

  @Test
  void empty_project_returns_zero_counts() {
    ProjectTree emptyTree = new ProjectTreeFile("/");
    NamespacePackageAnalyzer analyzer = new NamespacePackageAnalyzer();

    NamespacePackageTelemetry result = analyzer.analyze(emptyTree);

    assertThat(result.packagesWithInit()).isZero();
    assertThat(result.packagesWithoutInit()).isZero();
    assertThat(result.duplicatePackagesWithoutInit()).isZero();
    assertThat(result.namespacePackagesInRegularPackage()).isZero();
  }

  @Test
  void root_folder_is_excluded_from_counts() {
    ProjectTree rootWithPythonFile = new ProjectTreeFolder("/", List.of(
      new ProjectTreeFile("main.py")
    ));
    NamespacePackageAnalyzer analyzer = new NamespacePackageAnalyzer();

    NamespacePackageTelemetry result = analyzer.analyze(rootWithPythonFile);

    assertThat(result.packagesWithInit()).isZero();
    assertThat(result.packagesWithoutInit()).isZero();
  }

  @Test
  void folder_without_python_files_is_excluded() {
    ProjectTree tree = new ProjectTreeFolder("/", List.of(
      new ProjectTreeFolder("empty_folder", List.of()),
      new ProjectTreeFolder("config_folder", List.of(
        new ProjectTreeFile("config.json")
      ))
    ));
    NamespacePackageAnalyzer analyzer = new NamespacePackageAnalyzer();

    NamespacePackageTelemetry result = analyzer.analyze(tree);

    assertThat(result.packagesWithInit()).isZero();
    assertThat(result.packagesWithoutInit()).isZero();
  }

  @Test
  void package_with_init_is_counted() {
    ProjectTree tree = new ProjectTreeFolder("/", List.of(
      new ProjectTreeFolder("mypackage", List.of(
        new ProjectTreeFile("__init__.py"),
        new ProjectTreeFile("module.py")
      ))
    ));
    NamespacePackageAnalyzer analyzer = new NamespacePackageAnalyzer();

    NamespacePackageTelemetry result = analyzer.analyze(tree);

    assertThat(result.packagesWithInit()).isEqualTo(1);
    assertThat(result.packagesWithoutInit()).isZero();
    assertThat(result.duplicatePackagesWithoutInit()).isZero();
    assertThat(result.namespacePackagesInRegularPackage()).isZero();
  }

  @Test
  void package_without_init_is_counted() {
    ProjectTree tree = new ProjectTreeFolder("/", List.of(
      new ProjectTreeFolder("mypackage", List.of(
        new ProjectTreeFile("module.py")
      ))
    ));
    NamespacePackageAnalyzer analyzer = new NamespacePackageAnalyzer();

    NamespacePackageTelemetry result = analyzer.analyze(tree);

    assertThat(result.packagesWithInit()).isZero();
    assertThat(result.packagesWithoutInit()).isEqualTo(1);
    assertThat(result.duplicatePackagesWithoutInit()).isZero();
    assertThat(result.namespacePackagesInRegularPackage()).isZero();
  }

  @Test
  void namespace_package_is_detected_when_folder_appears_multiple_times() {
    ProjectTree tree = new ProjectTreeFolder("/", List.of(
      new ProjectTreeFolder("src1", List.of(
        new ProjectTreeFile("main1.py"),
        new ProjectTreeFolder("shared", List.of(
          new ProjectTreeFile("module1.py")
        ))
      )),
      new ProjectTreeFolder("src2", List.of(
        new ProjectTreeFile("main2.py"),
        new ProjectTreeFolder("shared", List.of(
          new ProjectTreeFile("module2.py")
        ))
      ))
    ));
    NamespacePackageAnalyzer analyzer = new NamespacePackageAnalyzer();

    NamespacePackageTelemetry result = analyzer.analyze(tree);

    assertThat(result.packagesWithoutInit()).isEqualTo(4);
    assertThat(result.duplicatePackagesWithoutInit()).isEqualTo(2);
  }

  @Test
  void package_missing_init_detected_when_parent_has_init() {
    ProjectTree tree = new ProjectTreeFolder("/", List.of(
      new ProjectTreeFolder("mypackage", List.of(
        new ProjectTreeFile("__init__.py"),
        new ProjectTreeFolder("subpackage", List.of(
          new ProjectTreeFile("module.py")
        ))
      ))
    ));
    NamespacePackageAnalyzer analyzer = new NamespacePackageAnalyzer();

    NamespacePackageTelemetry result = analyzer.analyze(tree);

    assertThat(result.packagesWithInit()).isEqualTo(1);
    assertThat(result.packagesWithoutInit()).isEqualTo(1);
    assertThat(result.namespacePackagesInRegularPackage()).isEqualTo(1);
  }

  @Test
  void package_missing_init_not_detected_when_no_parent_has_init() {
    ProjectTree tree = new ProjectTreeFolder("/", List.of(
      new ProjectTreeFolder("mypackage", List.of(
        new ProjectTreeFile("helper.py"),
        new ProjectTreeFolder("subpackage", List.of(
          new ProjectTreeFile("module.py")
        ))
      ))
    ));
    NamespacePackageAnalyzer analyzer = new NamespacePackageAnalyzer();

    NamespacePackageTelemetry result = analyzer.analyze(tree);

    assertThat(result.packagesWithInit()).isZero();
    assertThat(result.packagesWithoutInit()).isEqualTo(2);
    assertThat(result.namespacePackagesInRegularPackage()).isZero();
  }

  @Test
  void complex_project_structure() {
    ProjectTree tree = new ProjectTreeFolder("/", List.of(
      new ProjectTreeFolder("pkg1", List.of(
        new ProjectTreeFile("__init__.py"),
        new ProjectTreeFile("module1.py"),
        new ProjectTreeFolder("sub1", List.of(
          new ProjectTreeFile("__init__.py"),
          new ProjectTreeFile("module2.py")
        )),
        new ProjectTreeFolder("sub2", List.of(
          new ProjectTreeFile("module3.py")
        ))
      )),
      new ProjectTreeFolder("pkg2", List.of(
        new ProjectTreeFile("module4.py")
      )),
      new ProjectTreeFolder("src", List.of(
        new ProjectTreeFile("main.py"),
        new ProjectTreeFolder("shared", List.of(
          new ProjectTreeFile("util.py")
        ))
      )),
      new ProjectTreeFolder("lib", List.of(
        new ProjectTreeFile("app.py"),
        new ProjectTreeFolder("shared", List.of(
          new ProjectTreeFile("helper.py")
        ))
      ))
    ));
    NamespacePackageAnalyzer analyzer = new NamespacePackageAnalyzer();

    NamespacePackageTelemetry result = analyzer.analyze(tree);

    assertThat(result.packagesWithInit()).isEqualTo(2);
    assertThat(result.packagesWithoutInit()).isEqualTo(6);
    assertThat(result.duplicatePackagesWithoutInit()).isEqualTo(2);
    assertThat(result.namespacePackagesInRegularPackage()).isEqualTo(1);
  }

  @Test
  void deeply_nested_package_missing_init() {
    ProjectTree tree = new ProjectTreeFolder("/", List.of(
      new ProjectTreeFolder("top", List.of(
        new ProjectTreeFile("__init__.py"),
        new ProjectTreeFolder("level1", List.of(
          new ProjectTreeFile("__init__.py"),
          new ProjectTreeFolder("level2", List.of(
            new ProjectTreeFile("module.py")
          ))
        ))
      ))
    ));
    NamespacePackageAnalyzer analyzer = new NamespacePackageAnalyzer();

    NamespacePackageTelemetry result = analyzer.analyze(tree);

    assertThat(result.packagesWithInit()).isEqualTo(2);
    assertThat(result.packagesWithoutInit()).isEqualTo(1);
    assertThat(result.namespacePackagesInRegularPackage()).isEqualTo(1);
  }

  @Test
  void init_file_alone_without_other_python_files() {
    ProjectTree tree = new ProjectTreeFolder("/", List.of(
      new ProjectTreeFolder("pkg", List.of(
        new ProjectTreeFile("__init__.py")
      ))
    ));
    NamespacePackageAnalyzer analyzer = new NamespacePackageAnalyzer();

    NamespacePackageTelemetry result = analyzer.analyze(tree);

    assertThat(result.packagesWithInit()).isEqualTo(1);
    assertThat(result.packagesWithoutInit()).isZero();
  }

  @Test
  void telemetry_has_null_resolution_info_by_default() {
    ProjectTree emptyTree = new ProjectTreeFile("/");
    NamespacePackageAnalyzer analyzer = new NamespacePackageAnalyzer();

    NamespacePackageTelemetry result = analyzer.analyze(emptyTree);

    assertThat(result.resolutionMethod()).isNull();
    assertThat(result.buildSystem()).isNull();
  }

  @Test
  void telemetry_with_resolution_info_can_be_created() {
    ProjectTree emptyTree = new ProjectTreeFile("/");
    NamespacePackageAnalyzer analyzer = new NamespacePackageAnalyzer();
    NamespacePackageTelemetry baseTelemetry = analyzer.analyze(emptyTree);

    NamespacePackageTelemetry withResolution = baseTelemetry.withResolutionInfo(
      PackageResolutionResult.PrimaryResolutionMethod.PYPROJECT_TOML,
      PackageResolutionResult.BuildSystem.SETUPTOOLS);

    assertThat(withResolution.resolutionMethod()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.PYPROJECT_TOML);
    assertThat(withResolution.buildSystem()).isEqualTo(PackageResolutionResult.BuildSystem.SETUPTOOLS);
    // Original values preserved
    assertThat(withResolution.packagesWithInit()).isEqualTo(baseTelemetry.packagesWithInit());
    assertThat(withResolution.packagesWithoutInit()).isEqualTo(baseTelemetry.packagesWithoutInit());
  }

  @Test
  void telemetry_empty_has_null_resolution_info() {
    NamespacePackageTelemetry empty = NamespacePackageTelemetry.empty();

    assertThat(empty.resolutionMethod()).isNull();
    assertThat(empty.buildSystem()).isNull();
  }
}

