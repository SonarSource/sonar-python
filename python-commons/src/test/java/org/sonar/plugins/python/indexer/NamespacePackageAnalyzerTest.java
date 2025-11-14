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
package org.sonar.plugins.python.indexer;

import java.util.List;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class NamespacePackageAnalyzerTest {

  @Test
  void empty_project_returns_zero_counts() {
    ProjectTree emptyTree = new ProjectTree.ProjectTreeFile("/");
    NamespacePackageAnalyzer analyzer = new NamespacePackageAnalyzer();

    NamespacePackageTelemetry result = analyzer.analyze(emptyTree);

    assertThat(result.packagesWithInit()).isZero();
    assertThat(result.packagesWithoutInit()).isZero();
    assertThat(result.duplicatePackagesWithoutInit()).isZero();
    assertThat(result.namespacePackagesInRegularPackage()).isZero();
  }

  @Test
  void root_folder_is_excluded_from_counts() {
    ProjectTree rootWithPythonFile = new ProjectTree.ProjectTreeFolder("/", List.of(
      new ProjectTree.ProjectTreeFile("main.py")
    ));
    NamespacePackageAnalyzer analyzer = new NamespacePackageAnalyzer();

    NamespacePackageTelemetry result = analyzer.analyze(rootWithPythonFile);

    assertThat(result.packagesWithInit()).isZero();
    assertThat(result.packagesWithoutInit()).isZero();
  }

  @Test
  void folder_without_python_files_is_excluded() {
    ProjectTree tree = new ProjectTree.ProjectTreeFolder("/", List.of(
      new ProjectTree.ProjectTreeFolder("empty_folder", List.of()),
      new ProjectTree.ProjectTreeFolder("config_folder", List.of(
        new ProjectTree.ProjectTreeFile("config.json")
      ))
    ));
    NamespacePackageAnalyzer analyzer = new NamespacePackageAnalyzer();

    NamespacePackageTelemetry result = analyzer.analyze(tree);

    assertThat(result.packagesWithInit()).isZero();
    assertThat(result.packagesWithoutInit()).isZero();
  }

  @Test
  void package_with_init_is_counted() {
    ProjectTree tree = new ProjectTree.ProjectTreeFolder("/", List.of(
      new ProjectTree.ProjectTreeFolder("mypackage", List.of(
        new ProjectTree.ProjectTreeFile("__init__.py"),
        new ProjectTree.ProjectTreeFile("module.py")
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
    ProjectTree tree = new ProjectTree.ProjectTreeFolder("/", List.of(
      new ProjectTree.ProjectTreeFolder("mypackage", List.of(
        new ProjectTree.ProjectTreeFile("module.py")
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
    ProjectTree tree = new ProjectTree.ProjectTreeFolder("/", List.of(
      new ProjectTree.ProjectTreeFolder("src1", List.of(
        new ProjectTree.ProjectTreeFile("main1.py"),
        new ProjectTree.ProjectTreeFolder("shared", List.of(
          new ProjectTree.ProjectTreeFile("module1.py")
        ))
      )),
      new ProjectTree.ProjectTreeFolder("src2", List.of(
        new ProjectTree.ProjectTreeFile("main2.py"),
        new ProjectTree.ProjectTreeFolder("shared", List.of(
          new ProjectTree.ProjectTreeFile("module2.py")
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
    ProjectTree tree = new ProjectTree.ProjectTreeFolder("/", List.of(
      new ProjectTree.ProjectTreeFolder("mypackage", List.of(
        new ProjectTree.ProjectTreeFile("__init__.py"),
        new ProjectTree.ProjectTreeFolder("subpackage", List.of(
          new ProjectTree.ProjectTreeFile("module.py")
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
    ProjectTree tree = new ProjectTree.ProjectTreeFolder("/", List.of(
      new ProjectTree.ProjectTreeFolder("mypackage", List.of(
        new ProjectTree.ProjectTreeFile("helper.py"),
        new ProjectTree.ProjectTreeFolder("subpackage", List.of(
          new ProjectTree.ProjectTreeFile("module.py")
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
    ProjectTree tree = new ProjectTree.ProjectTreeFolder("/", List.of(
      new ProjectTree.ProjectTreeFolder("pkg1", List.of(
        new ProjectTree.ProjectTreeFile("__init__.py"),
        new ProjectTree.ProjectTreeFile("module1.py"),
        new ProjectTree.ProjectTreeFolder("sub1", List.of(
          new ProjectTree.ProjectTreeFile("__init__.py"),
          new ProjectTree.ProjectTreeFile("module2.py")
        )),
        new ProjectTree.ProjectTreeFolder("sub2", List.of(
          new ProjectTree.ProjectTreeFile("module3.py")
        ))
      )),
      new ProjectTree.ProjectTreeFolder("pkg2", List.of(
        new ProjectTree.ProjectTreeFile("module4.py")
      )),
      new ProjectTree.ProjectTreeFolder("src", List.of(
        new ProjectTree.ProjectTreeFile("main.py"),
        new ProjectTree.ProjectTreeFolder("shared", List.of(
          new ProjectTree.ProjectTreeFile("util.py")
        ))
      )),
      new ProjectTree.ProjectTreeFolder("lib", List.of(
        new ProjectTree.ProjectTreeFile("app.py"),
        new ProjectTree.ProjectTreeFolder("shared", List.of(
          new ProjectTree.ProjectTreeFile("helper.py")
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
    ProjectTree tree = new ProjectTree.ProjectTreeFolder("/", List.of(
      new ProjectTree.ProjectTreeFolder("top", List.of(
        new ProjectTree.ProjectTreeFile("__init__.py"),
        new ProjectTree.ProjectTreeFolder("level1", List.of(
          new ProjectTree.ProjectTreeFile("__init__.py"),
          new ProjectTree.ProjectTreeFolder("level2", List.of(
            new ProjectTree.ProjectTreeFile("module.py")
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
    ProjectTree tree = new ProjectTree.ProjectTreeFolder("/", List.of(
      new ProjectTree.ProjectTreeFolder("pkg", List.of(
        new ProjectTree.ProjectTreeFile("__init__.py")
      ))
    ));
    NamespacePackageAnalyzer analyzer = new NamespacePackageAnalyzer();

    NamespacePackageTelemetry result = analyzer.analyze(tree);

    assertThat(result.packagesWithInit()).isEqualTo(1);
    assertThat(result.packagesWithoutInit()).isZero();
  }
}

