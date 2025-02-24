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

import java.util.Collection;
import java.util.stream.Collectors;
import org.assertj.core.api.AbstractCollectionAssert;
import org.assertj.core.api.Assertions;
import org.assertj.core.api.ObjectAssert;
import org.junit.jupiter.api.Test;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.plugins.python.dependency.model.Dependencies;
import org.sonar.plugins.python.dependency.model.Dependency;

import static org.assertj.core.api.Assertions.as;
import static org.assertj.core.api.InstanceOfAssertFactories.SET;

class PyProjectTomlParserTest {

  @Test
  void testInvalidTomlFile() {
    assertThat("[project").isEmpty();
    assertThat("""
      [project]
      dependencies = ["httpx"]
      dependencies = []
      """).isEmpty();
  }

  @Test
  void testWithEmptyFile() {
    assertThat("").isEmpty();
  }

  @Test
  void testWithEmptyProject() {
    assertThat("""
      [project]
      """).isEmpty();
  }

  @Test
  void testEmptyVersion() {
    assertThat("""
      [project]
      dependencies = [""]
      """).isEmpty();
  }

  @Test
  void testWithDependency() {
    assertThat("""
      [project]
      dependencies = [
        "httpx",
        "gidgethub[httpx]>4.0.0",
        "django>2.1; os_name != 'nt'",
        "django>2.0; os_name == 'nt'",
        "pip @ https://github.com/pypa/pip/archive/1.3.1.zip#sha1=da9234ee9982d4bbb3c72346a6de940a148ea686",
        "test_package1",
        "test-package2",
        "test.package3",
      ]
      """).containsExactlyInAnyOrder("httpx", "gidgethub", "django", "pip", "test-package1", "test-package2", "test-package3");

    assertThat("""
      [project]
      dependencies = []
      """).isEmpty();
  }

  @Test
  void testWithPoetryStyleDependencies() {
    assertThat("""
      [tool.poetry.dependencies]
      package1 = "1.9"
      package2 = "^2.13.0"
      package3 = ">=3.8,<3.9.7 || >3.9.7,<4.0"
      """).containsExactlyInAnyOrder("package1", "package2", "package3");
  }

  @Test
  void testIllFormedPoetryStyleDependencies() {
    assertThat("""
      [tool]
      smthWhichIsNotAPoetryDependency = "1.9"
      """).isEmpty();

    assertThat("""
      [tool.poetry]
      smthWhichIsNotPoetryDependency = "1.9"
      """).isEmpty();
  }

  @Test
  void testCompleteTomlFile() {
    assertThat("""
      [project]
      name = "test"
      version = "0.1.0"
      description = "Add your description here"
      readme = "README.md"
      requires-python = ">=3.13"
      dependencies = [
          "pytest-cov>=6.0.0",
      ]

      [tool.poetry.dependencies]
      requests = "1.9"
      """).containsExactlyInAnyOrder("pytest-cov", "requests");
  }

  private static AbstractCollectionAssert<?, Collection<?>, Object, ObjectAssert<Object>> assertThat(String code) {
    return Assertions.assertThat(parse(code)).extracting(
      dependencies -> dependencies.dependencies().stream().map(Dependency::name).collect(Collectors.toSet()),
      as(SET));
  }

  private static Dependencies parse(String content) {
    var inputFile = TestInputFileBuilder.create("modulekey", "pyproject.toml")
      .setContents(content)
      .build();
    return PyProjectTomlParser.parse(inputFile);
  }
}
