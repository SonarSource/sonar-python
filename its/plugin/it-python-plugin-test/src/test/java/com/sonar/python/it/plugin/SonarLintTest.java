/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package com.sonar.python.it.plugin;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.sonar.api.batch.fs.InputFile;
import org.sonarsource.sonarlint.core.StandaloneSonarLintEngineImpl;
import org.sonarsource.sonarlint.core.analysis.api.ClientInputFile;
import org.sonarsource.sonarlint.core.analysis.api.ClientModuleFileSystem;
import org.sonarsource.sonarlint.core.analysis.api.ClientModuleInfo;
import org.sonarsource.sonarlint.core.client.api.common.analysis.Issue;
import org.sonarsource.sonarlint.core.client.api.standalone.StandaloneAnalysisConfiguration;
import org.sonarsource.sonarlint.core.client.api.standalone.StandaloneGlobalConfiguration;
import org.sonarsource.sonarlint.core.client.api.standalone.StandaloneSonarLintEngine;
import org.sonarsource.sonarlint.core.commons.IssueSeverity;
import org.sonarsource.sonarlint.core.commons.Language;
import org.sonarsource.sonarlint.core.commons.log.ClientLogOutput;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.tuple;

class SonarLintTest {

  @TempDir
  public static Path TEMP;

  private static StandaloneSonarLintEngine sonarlintEngine;

  private static File baseDir;

  @BeforeAll
  static void prepare() throws Exception {
    StandaloneGlobalConfiguration sonarLintConfig = StandaloneGlobalConfiguration.builder()
      .addPlugin(TestsUtils.PLUGIN_LOCATION.getFile().toPath())
      .setSonarLintUserHome(TEMP)
      .addEnabledLanguage(Language.PYTHON)
      .setLogOutput((formattedMessage, level) -> {
        /* Don't pollute logs */ })
      .setModulesProvider(Collections::emptyList)
      .build();
    sonarlintEngine = new StandaloneSonarLintEngineImpl(sonarLintConfig);
    baseDir = TEMP.toFile();
  }

  @AfterAll
  static void stop() {
    sonarlintEngine.stop();
  }

  @Test
  void should_raise_issues() throws IOException {
    ClientInputFile inputFile = prepareInputFile("foo.py",
      "def fooBar():\n"
        + "  `1` \n"
        + "  `1` #NOSONAR\n",
      false);

    List<Issue> issues = new ArrayList<>();
    StandaloneAnalysisConfiguration configuration =
      StandaloneAnalysisConfiguration.builder()
        .setBaseDir(baseDir.toPath())
        .addInputFile(inputFile)
        .setModuleKey("myModule")
        .build();

    Map<ClientLogOutput.Level, List<String>> logsByLevel = new HashMap<>();
    ClientLogOutput logOutput = (s, level) -> {
      List<String> logs = logsByLevel.getOrDefault(level, new ArrayList<>());
      logs.add(s);
      logsByLevel.putIfAbsent(level, logs);
    };
    ClientModuleFileSystem clientFileSystem = new ClientModuleFileSystem() {
      @Override
      public Stream<ClientInputFile> files(String s, InputFile.Type type) {
        return Stream.of(inputFile);
      }

      @Override
      public Stream<ClientInputFile> files() {
        return Stream.of(inputFile);
      }
    };
    sonarlintEngine.declareModule(new ClientModuleInfo("myModule", clientFileSystem));
    sonarlintEngine.analyze(configuration, issues::add, logOutput, null);

    assertThat(logsByLevel.get(ClientLogOutput.Level.WARN)).containsExactly("No workDir in SonarLint");
    assertThat(issues).extracting("ruleKey", "startLine", "inputFile.path", "severity").containsOnly(
      tuple("python:BackticksUsage", 2, inputFile.uri().getPath(), IssueSeverity.BLOCKER),
      tuple("python:S1542", 1, inputFile.uri().getPath(), IssueSeverity.MAJOR));
  }

  private static ClientInputFile prepareInputFile(String relativePath, String content, final boolean isTest) throws IOException {
    File file = new File(baseDir, relativePath);
    FileUtils.write(file, content, StandardCharsets.UTF_8);
    return createInputFile(file.toPath(), relativePath, isTest);
  }

  private static ClientInputFile createInputFile(final Path path, String relativePath, final boolean isTest) {
    return new ClientInputFile() {

      @Override
      public String getPath() {
        return path.toString();
      }

      @Override
      public boolean isTest() {
        return isTest;
      }

      @Override
      public Charset getCharset() {
        return StandardCharsets.UTF_8;
      }

      @Override
      public <G> G getClientObject() {
        return null;
      }

      @Override
      public InputStream inputStream() throws IOException {
        return Files.newInputStream(path);
      }

      @Override
      public String contents() throws IOException {
        return new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
      }

      @Override
      public String relativePath() {
        return relativePath;
      }

      @Override
      public URI uri() {
        return path.toUri();
      }

    };
  }
}
