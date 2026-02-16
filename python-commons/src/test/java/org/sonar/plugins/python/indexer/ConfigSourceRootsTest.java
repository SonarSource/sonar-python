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

class ConfigSourceRootsTest {

  @TempDir
  Path tempDir;

  @Test
  void toAbsolutePaths_resolvesRelativeToConfigFileParent() throws IOException {
    Path subDir = tempDir.resolve("subproject");
    Files.createDirectories(subDir);
    File configFile = subDir.resolve("pyproject.toml").toFile();
    Files.writeString(configFile.toPath(), "");

    ConfigSourceRoots csr = new ConfigSourceRoots(configFile, List.of("src", "lib"));

    assertThat(csr.toAbsolutePaths()).containsExactly(
      subDir.resolve("src").toFile().getAbsolutePath(),
      subDir.resolve("lib").toFile().getAbsolutePath()
    );
  }

  @Test
  void toAbsolutePaths_emptyListForEmptyRelativeRoots() throws IOException {
    File configFile = tempDir.resolve("pyproject.toml").toFile();
    Files.writeString(configFile.toPath(), "");

    ConfigSourceRoots csr = new ConfigSourceRoots(configFile, List.of());

    assertThat(csr.toAbsolutePaths()).isEmpty();
  }

  @Test
  void toAbsolutePaths_handlesNestedDirectories() throws IOException {
    Path deepDir = tempDir.resolve("a/b/c");
    Files.createDirectories(deepDir);
    File configFile = deepDir.resolve("pyproject.toml").toFile();
    Files.writeString(configFile.toPath(), "");

    ConfigSourceRoots csr = new ConfigSourceRoots(configFile, List.of("src"));

    assertThat(csr.toAbsolutePaths()).containsExactly(
      deepDir.resolve("src").toFile().getAbsolutePath()
    );
  }

  @Test
  void empty_createsConfigSourceRootsWithEmptyRelativeRoots() throws IOException {
    File configFile = tempDir.resolve("pyproject.toml").toFile();
    Files.writeString(configFile.toPath(), "");

    ConfigSourceRoots csr = ConfigSourceRoots.empty(configFile);

    assertThat(csr.configFile()).isEqualTo(configFile);
    assertThat(csr.relativeRoots()).isEmpty();
    assertThat(csr.toAbsolutePaths()).isEmpty();
  }

  @Test
  void configFile_returnsOriginalFile() throws IOException {
    File configFile = tempDir.resolve("setup.py").toFile();
    Files.writeString(configFile.toPath(), "");

    ConfigSourceRoots csr = new ConfigSourceRoots(configFile, List.of("src"));

    assertThat(csr.configFile()).isSameAs(configFile);
  }

  @Test
  void relativeRoots_returnsOriginalList() throws IOException {
    File configFile = tempDir.resolve("pyproject.toml").toFile();
    Files.writeString(configFile.toPath(), "");
    List<String> roots = List.of("src", "lib");

    ConfigSourceRoots csr = new ConfigSourceRoots(configFile, roots);

    assertThat(csr.relativeRoots()).isEqualTo(roots);
  }

  @Test
  void toAbsolutePaths_returnsEmptyListWhenParentDirIsNull() {
    // A File with no parent (relative path with no directory component)
    File configFile = new File("pyproject.toml");

    ConfigSourceRoots csr = new ConfigSourceRoots(configFile, List.of("src", "lib"));

    assertThat(csr.toAbsolutePaths()).isEmpty();
  }
}
