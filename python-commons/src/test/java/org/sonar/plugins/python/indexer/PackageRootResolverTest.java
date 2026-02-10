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
import org.sonar.api.config.Configuration;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PackageRootResolverTest {

  @TempDir
  Path tempDir;

  @Test
  void resolve_withExtractedRoots_returnsAbsolutePaths() {
    Configuration config = mock(Configuration.class);
    when(config.getStringArray("sonar.sources")).thenReturn(new String[0]);

    File baseDir = tempDir.toFile();
    List<String> extractedRoots = List.of("src", "lib");
    List<String> result = PackageRootResolver.resolve(extractedRoots, config, baseDir);

    assertThat(result).containsExactly(
      new File(baseDir, "src").getAbsolutePath(),
      new File(baseDir, "lib").getAbsolutePath());
  }

  @Test
  void resolve_withExtractedRoots_ignoresSonarSources() {
    Configuration config = mock(Configuration.class);
    when(config.getStringArray("sonar.sources")).thenReturn(new String[]{"other"});

    File baseDir = tempDir.toFile();
    List<String> extractedRoots = List.of("src");
    List<String> result = PackageRootResolver.resolve(extractedRoots, config, baseDir);

    assertThat(result).containsExactly(new File(baseDir, "src").getAbsolutePath());
  }

  @Test
  void resolve_emptyExtractedRoots_fallsBackToSonarSources() {
    Configuration config = mock(Configuration.class);
    when(config.getStringArray("sonar.sources")).thenReturn(new String[]{"sources", "lib"});

    File baseDir = tempDir.toFile();
    List<String> result = PackageRootResolver.resolve(List.of(), config, baseDir);

    assertThat(result).containsExactly(
      new File(baseDir, "sources").getAbsolutePath(),
      new File(baseDir, "lib").getAbsolutePath());
  }

  @Test
  void resolve_emptyExtractedRootsAndNoSonarSources_fallsBackToSrcFolder() throws IOException {
    Configuration config = mock(Configuration.class);
    when(config.getStringArray("sonar.sources")).thenReturn(new String[0]);

    File baseDir = tempDir.toFile();
    Files.createDirectory(tempDir.resolve("src"));

    List<String> result = PackageRootResolver.resolve(List.of(), config, baseDir);

    assertThat(result).containsExactly(new File(baseDir, "src").getAbsolutePath());
  }

  @Test
  void resolve_emptyExtractedRootsAndNoSonarSources_fallsBackToLibFolder() throws IOException {
    Configuration config = mock(Configuration.class);
    when(config.getStringArray("sonar.sources")).thenReturn(new String[0]);

    File baseDir = tempDir.toFile();
    Files.createDirectory(tempDir.resolve("lib"));

    List<String> result = PackageRootResolver.resolve(List.of(), config, baseDir);

    assertThat(result).containsExactly(new File(baseDir, "lib").getAbsolutePath());
  }

  @Test
  void resolve_emptyExtractedRootsAndNoSonarSources_fallsBackToBothSrcAndLib() throws IOException {
    Configuration config = mock(Configuration.class);
    when(config.getStringArray("sonar.sources")).thenReturn(new String[0]);

    File baseDir = tempDir.toFile();
    Files.createDirectory(tempDir.resolve("src"));
    Files.createDirectory(tempDir.resolve("lib"));

    List<String> result = PackageRootResolver.resolve(List.of(), config, baseDir);

    assertThat(result).containsExactly(
      new File(baseDir, "src").getAbsolutePath(),
      new File(baseDir, "lib").getAbsolutePath());
  }

  @Test
  void resolve_emptyExtractedRootsNoSonarSourcesNoSrcFolder_fallsBackToBaseDirAbsolutePath() {
    Configuration config = mock(Configuration.class);
    when(config.getStringArray("sonar.sources")).thenReturn(new String[0]);

    File baseDir = tempDir.toFile();
    List<String> result = PackageRootResolver.resolve(List.of(), config, baseDir);

    assertThat(result).containsExactly(baseDir.getAbsolutePath());
  }

  @Test
  void resolve_srcExistsAsFile_fallsBackToBaseDirAbsolutePath() throws IOException {
    Configuration config = mock(Configuration.class);
    when(config.getStringArray("sonar.sources")).thenReturn(new String[0]);

    Files.createFile(tempDir.resolve("src"));

    File baseDir = tempDir.toFile();
    List<String> result = PackageRootResolver.resolve(List.of(), config, baseDir);

    assertThat(result).containsExactly(baseDir.getAbsolutePath());
  }

  // Tests for resolveFallback method directly

  @Test
  void resolveFallback_withSonarSources_returnsAbsolutePaths() {
    Configuration config = mock(Configuration.class);
    when(config.getStringArray("sonar.sources")).thenReturn(new String[]{"app", "core"});

    File baseDir = tempDir.toFile();
    List<String> result = PackageRootResolver.resolveFallback(config, baseDir);

    assertThat(result).containsExactly(
      new File(baseDir, "app").getAbsolutePath(),
      new File(baseDir, "core").getAbsolutePath());
  }

  @Test
  void resolveFallback_noSonarSourcesButSrcExists_returnsAbsolutePath() throws IOException {
    Configuration config = mock(Configuration.class);
    when(config.getStringArray("sonar.sources")).thenReturn(new String[0]);

    File baseDir = tempDir.toFile();
    Files.createDirectory(tempDir.resolve("src"));

    List<String> result = PackageRootResolver.resolveFallback(config, baseDir);

    assertThat(result).containsExactly(new File(baseDir, "src").getAbsolutePath());
  }

  @Test
  void resolveFallback_noSonarSourcesButLibExists_returnsAbsolutePath() throws IOException {
    Configuration config = mock(Configuration.class);
    when(config.getStringArray("sonar.sources")).thenReturn(new String[0]);

    File baseDir = tempDir.toFile();
    Files.createDirectory(tempDir.resolve("lib"));

    List<String> result = PackageRootResolver.resolveFallback(config, baseDir);

    assertThat(result).containsExactly(new File(baseDir, "lib").getAbsolutePath());
  }

  @Test
  void resolveFallback_noSonarSourcesBothSrcAndLibExist_returnsAbsolutePaths() throws IOException {
    Configuration config = mock(Configuration.class);
    when(config.getStringArray("sonar.sources")).thenReturn(new String[0]);

    File baseDir = tempDir.toFile();
    Files.createDirectory(tempDir.resolve("src"));
    Files.createDirectory(tempDir.resolve("lib"));

    List<String> result = PackageRootResolver.resolveFallback(config, baseDir);

    assertThat(result).containsExactly(
      new File(baseDir, "src").getAbsolutePath(),
      new File(baseDir, "lib").getAbsolutePath());
  }

  @Test
  void resolveFallback_noSonarSourcesNoSrcNoLib_returnsBaseDirAbsolutePath() {
    Configuration config = mock(Configuration.class);
    when(config.getStringArray("sonar.sources")).thenReturn(new String[0]);

    File baseDir = tempDir.toFile();
    List<String> result = PackageRootResolver.resolveFallback(config, baseDir);

    assertThat(result).containsExactly(baseDir.getAbsolutePath());
  }

  @Test
  void resolveFallback_srcAndLibExistAsFiles_returnsBaseDirAbsolutePath() throws IOException {
    Configuration config = mock(Configuration.class);
    when(config.getStringArray("sonar.sources")).thenReturn(new String[0]);

    Files.createFile(tempDir.resolve("src"));
    Files.createFile(tempDir.resolve("lib"));

    File baseDir = tempDir.toFile();
    List<String> result = PackageRootResolver.resolveFallback(config, baseDir);

    assertThat(result).containsExactly(baseDir.getAbsolutePath());
  }
}

