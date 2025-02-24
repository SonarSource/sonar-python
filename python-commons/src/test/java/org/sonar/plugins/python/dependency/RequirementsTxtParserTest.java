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
import org.junit.jupiter.api.Test;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.plugins.python.TestUtils;
import org.sonar.plugins.python.dependency.model.Dependencies;
import org.sonar.plugins.python.dependency.model.Dependency;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

class RequirementsTxtParserTest {

  private final Path baseDir = Path.of("src/test/resources/org/sonar/plugins/python/sensor");

  @Test
  void parse_error_on_requirement_files() throws IOException {
    DefaultInputFile inputFile = spy(createRequirementFile());
    when(inputFile.contents()).thenThrow(FileNotFoundException.class);
    Dependencies dependencies = RequirementsTxtParser.parseRequirementFile(inputFile);
    assertThat(dependencies.dependencies()).isEmpty();
  }

  @Test
  void parse_requirement_files() {
    DefaultInputFile requirementFile = createRequirementFile();
    Dependencies dependencies = RequirementsTxtParser.parseRequirementFile(requirementFile);
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
      "packagewithvcs");
  }

  private DefaultInputFile createRequirementFile() {
    String name = "requirements.txt";
    return TestInputFileBuilder.create("moduleKey", baseDir.resolve(name).toString())
      .setModuleBaseDir(baseDir)
      .setCharset(UTF_8)
      .setContents(TestUtils.fileContent(new File(baseDir.toFile(), name), UTF_8))
      .build();
  }
}
