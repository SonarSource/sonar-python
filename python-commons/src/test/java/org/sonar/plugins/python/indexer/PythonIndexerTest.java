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
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.project.config.ProjectConfigurationBuilder;
import org.sonar.python.semantic.v2.typetable.CachedTypeTable;
import org.sonar.python.semantic.v2.typetable.TypeTable;

import static org.assertj.core.api.Assertions.assertThat;

class PythonIndexerTest {

  @Test
  void test_projectLevelTypeTableIsCached() {
    TestPythonIndexer pythonIndexer = new TestPythonIndexer(new ProjectConfigurationBuilder());
    TypeTable typeTable = pythonIndexer.projectLevelTypeTable();
    assertThat(typeTable).isInstanceOf(CachedTypeTable.class);
  }

  @Test
  void test_adjustPackageRoot_noInitPy(@TempDir Path tempDir) {
    File baseDir = tempDir.toFile();
    File srcDir = new File(baseDir, "src");
    srcDir.mkdir();

    String result = PythonIndexer.adjustPackageRoot(srcDir, baseDir);
    assertThat(result).isEqualTo(srcDir.getAbsolutePath());
  }

  @Test
  void test_adjustPackageRoot_withInitPy_walksUp(@TempDir Path tempDir) throws Exception {
    File baseDir = tempDir.toFile();
    File srcDir = new File(baseDir, "src");
    File packageDir = new File(srcDir, "mypackage");
    packageDir.mkdirs();
    new File(packageDir, "__init__.py").createNewFile();

    String result = PythonIndexer.adjustPackageRoot(packageDir, baseDir);
    assertThat(result).isEqualTo(srcDir.getAbsolutePath());
  }

  @Test
  void test_adjustPackageRoot_nestedPackagesWithInitPy(@TempDir Path tempDir) throws Exception {
    File baseDir = tempDir.toFile();
    File srcDir = new File(baseDir, "src");
    File level1 = new File(srcDir, "level1");
    File level2 = new File(level1, "level2");
    File level3 = new File(level2, "level3");
    level3.mkdirs();

    new File(level1, "__init__.py").createNewFile();
    new File(level2, "__init__.py").createNewFile();
    new File(level3, "__init__.py").createNewFile();

    String result = PythonIndexer.adjustPackageRoot(level3, baseDir);
    assertThat(result).isEqualTo(srcDir.getAbsolutePath());
  }

  @Test
  void test_adjustPackageRoot_stopsAtBaseDir(@TempDir Path tempDir) throws Exception {
    File baseDir = tempDir.toFile();
    File level1 = new File(baseDir, "level1");
    File level2 = new File(level1, "level2");
    level2.mkdirs();
    new File(baseDir, "__init__.py").createNewFile();
    new File(level1, "__init__.py").createNewFile();
    new File(level2, "__init__.py").createNewFile();

    String result = PythonIndexer.adjustPackageRoot(level2, baseDir);
    assertThat(result).isEqualTo(baseDir.getAbsolutePath());
  }

  @Test
  void test_adjustPackageRoot_rootEqualsBaseDir(@TempDir Path tempDir) throws Exception {
    File baseDir = tempDir.toFile();
    new File(baseDir, "__init__.py").createNewFile();

    String result = PythonIndexer.adjustPackageRoot(baseDir, baseDir);
    assertThat(result).isEqualTo(baseDir.getAbsolutePath());
  }

  @Test
  void test_adjustPackageRoot_partialInitPyChain(@TempDir Path tempDir) throws Exception {
    File baseDir = tempDir.toFile();
    File srcDir = new File(baseDir, "src");
    File withInit = new File(srcDir, "withInit");
    File withoutInit = new File(withInit, "withoutInit");
    File deepPackage = new File(withoutInit, "deepPackage");
    deepPackage.mkdirs();

    new File(withInit, "__init__.py").createNewFile();
    new File(deepPackage, "__init__.py").createNewFile();

    String result = PythonIndexer.adjustPackageRoot(deepPackage, baseDir);
    assertThat(result).isEqualTo(withoutInit.getAbsolutePath());
  }

  @Test
  void test_adjustPackageRoot_emptyDirectory(@TempDir Path tempDir) {
    File baseDir = tempDir.toFile();
    File emptyDir = new File(baseDir, "empty");
    emptyDir.mkdir();

    String result = PythonIndexer.adjustPackageRoot(emptyDir, baseDir);
    assertThat(result).isEqualTo(emptyDir.getAbsolutePath());
  }

  @Test
  void test_adjustPackageRoot_singleLevelWithInitPy(@TempDir Path tempDir) throws Exception {
    File baseDir = tempDir.toFile();
    File packageDir = new File(baseDir, "mypackage");
    packageDir.mkdir();
    new File(packageDir, "__init__.py").createNewFile();

    String result = PythonIndexer.adjustPackageRoot(packageDir, baseDir);
    assertThat(result).isEqualTo(baseDir.getAbsolutePath());
  }

  private static class TestPythonIndexer extends PythonIndexer {

    protected TestPythonIndexer(ProjectConfigurationBuilder projectConfigurationBuilder) {
      super(projectConfigurationBuilder);
    }

    @Override
    public void buildOnce(SensorContext context) {
      // do nothing
    }

    @Override
    public void postAnalysis(SensorContext context) {
      // do nothing
    }

    @Override
    public CacheContext cacheContext() {
      return CacheContextImpl.dummyCache();
    }
  }
}
