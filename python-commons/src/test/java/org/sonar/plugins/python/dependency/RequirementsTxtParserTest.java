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

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.api.batch.fs.IndexedFile;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.plugins.python.TestUtils;
import org.sonar.plugins.python.dependency.model.Dependencies;
import org.sonar.plugins.python.dependency.model.Dependency;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

class RequirementsTxtParserTest {

  private final File baseDir = new File("src/test/resources/org/sonar/plugins/python/sensor").getAbsoluteFile();

  private SensorContextTester context = SensorContextTester.create(baseDir);

  @Test
  void list_requirement_files() {
    requirementFile();
    requirementFile(Path.of("subdir"));
    List<InputFile> requirements = RequirementsTxtParser.getRequirementFiles(context);
    assertThat(requirements).hasSize(2);
    assertThat(requirements.stream().map(IndexedFile::filename).toList()).allMatch(f -> f.equals("requirements.txt"));
  }

  @Test
  void parse_error_on_requirement_files() throws IOException {
    DefaultInputFile inputFile = spy(createRequirementFile(Path.of(".")));
    when(inputFile.contents()).thenThrow(FileNotFoundException.class);
    context.fileSystem().add(inputFile);
    Dependencies dependencies = RequirementsTxtParser.parseRequirementFiles(context);
    assertThat(dependencies.dependencies()).isEmpty();
  }

  @Test
  void parse_requirement_files() {
    requirementFile();
    requirementFile(Path.of("subdir"));
    Dependencies dependencies = RequirementsTxtParser.parseRequirementFiles(context);
    assertThat(dependencies.dependencies()).map(Dependency::name).containsExactlyInAnyOrder(
      "package1",
      "package2",
      "package-with-dash",
      "package-with-dot",
      "package-with-underscore",
      "packagewithcomment",
      "packagewithleadingspaces",
      "packagewithfixedversion",
      "packagewithconstraints",
      "packagewithvcs",
      "packagefromsubdir");
  }

  private void requirementFile() {
    requirementFile(Path.of("."));
  }

  private void requirementFile(Path relativePath) {
    DefaultInputFile requirementFile = createRequirementFile(relativePath);
    context.fileSystem().add(requirementFile);
  }

  private DefaultInputFile createRequirementFile(Path relativePath) {
    String name = "requirements.txt";
    Path dir = baseDir.toPath().resolve(relativePath);
    return TestInputFileBuilder.create("moduleKey", relativePath.resolve(name).toString())
      .setModuleBaseDir(baseDir.toPath().resolve(relativePath))
      .setCharset(UTF_8)
      .setContents(TestUtils.fileContent(new File(dir.toFile(), name), UTF_8))
      .build();
  }
}
