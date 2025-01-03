/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2025 SonarSource SA
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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.sonar.api.batch.fs.InputFile;
import org.sonarsource.sonarlint.core.analysis.AnalysisEngine;
import org.sonarsource.sonarlint.core.analysis.api.ActiveRule;
import org.sonarsource.sonarlint.core.analysis.api.AnalysisConfiguration;
import org.sonarsource.sonarlint.core.analysis.api.AnalysisEngineConfiguration;
import org.sonarsource.sonarlint.core.analysis.api.ClientInputFile;
import org.sonarsource.sonarlint.core.analysis.api.ClientModuleFileSystem;
import org.sonarsource.sonarlint.core.analysis.api.ClientModuleInfo;
import org.sonarsource.sonarlint.core.analysis.api.Issue;
import org.sonarsource.sonarlint.core.analysis.command.AnalyzeCommand;
import org.sonarsource.sonarlint.core.analysis.command.RegisterModuleCommand;
import org.sonarsource.sonarlint.core.commons.api.SonarLanguage;
import org.sonarsource.sonarlint.core.commons.log.LogOutput;
import org.sonarsource.sonarlint.core.commons.log.LogOutput.Level;
import org.sonarsource.sonarlint.core.commons.log.SonarLintLogger;
import org.sonarsource.sonarlint.core.commons.progress.ProgressMonitor;
import org.sonarsource.sonarlint.core.plugin.commons.PluginsLoader;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.tuple;

class SonarLintTest {

  @TempDir
  public static Path temp;

  private static AnalysisEngine sonarlintEngine;
  private final ProgressMonitor progressMonitor = new ProgressMonitor(null);

  private static File baseDir;

  @BeforeAll
  static void prepare() {
    var sonarLintConfig = AnalysisEngineConfiguration.builder()
      .setWorkDir(temp)
      .build();

    var logOutput = new LogOutput() {
      @Override
      public void log(String formattedMessage, Level level, @Nullable String stacktrace) {
        /* Don't pollute logs */
      }
    };
    SonarLintLogger.setTarget(logOutput);
    var pluginJarLocation = Set.of(TestsUtils.PLUGIN_LOCATION.getFile().toPath());
    var enabledLanguages = Set.of(SonarLanguage.PYTHON);
    var pluginConfiguration = new PluginsLoader.Configuration(pluginJarLocation, enabledLanguages, false, Optional.empty());
    var pluginLoader = new PluginsLoader().load(pluginConfiguration, Set.of());

    sonarlintEngine = new AnalysisEngine(sonarLintConfig, pluginLoader.getLoadedPlugins(), logOutput);
    baseDir = temp.toFile();
  }

  @AfterAll
  static void stop() {
    SonarLintLogger.setTarget(null);
    sonarlintEngine.stop();
  }

  @Test
  void should_raise_issues() throws IOException, InterruptedException, ExecutionException {
    ClientInputFile inputFile = prepareInputFile("foo.py",
      "def fooBar():\n"
        + "  `1` \n"
        + "  `1` #NOSONAR\n",
      false);

    List<Issue> issues = new ArrayList<>();
    var configuration = AnalysisConfiguration.builder()
      .setBaseDir(baseDir.toPath())
      .addInputFile(inputFile)
      .addActiveRules(
        new ActiveRule("python:S1542", SonarLanguage.PYTHON.name()),
        new ActiveRule("python:BackticksUsage", SonarLanguage.PYTHON.name()))
      .build();

    Map<Level, List<String>> logsByLevel = new HashMap<>();
    LogOutput logOutput = new LogOutput() {
      @Override
      public void log(String formattedMessage, Level level, @Nullable String stacktrace) {
        logsByLevel.computeIfAbsent(level, k -> new ArrayList<>()).add(formattedMessage);
      }
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
    sonarlintEngine.post(new RegisterModuleCommand(new ClientModuleInfo("myModule", clientFileSystem)), progressMonitor).get();
    var command = new AnalyzeCommand("myModule", configuration, issues::add, logOutput);
    sonarlintEngine.post(command, progressMonitor).get();

    assertThat(logsByLevel.get(Level.WARN)).containsExactly("No workDir in SonarLint");
    assertThat(issues).extracting("ruleKey", "textRange.startLine", "inputFile.path").containsOnly(
      tuple("python:BackticksUsage", 2, inputFile.uri().getPath()),
      tuple("python:S1542", 1, inputFile.uri().getPath()));
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
