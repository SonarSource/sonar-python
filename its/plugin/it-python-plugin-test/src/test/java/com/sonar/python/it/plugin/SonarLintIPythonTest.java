/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2024 SonarSource SA
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
import org.sonarsource.sonarlint.core.analysis.api.WithTextRange;
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

class SonarLintIPythonTest {

  @TempDir
  public static Path temp;

  private static AnalysisEngine sonarlintEngine;
  private final ProgressMonitor progressMonitor = new ProgressMonitor(null);

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
    var enabledLanguages = Set.of(SonarLanguage.IPYTHON);
    var pluginConfiguration = new PluginsLoader.Configuration(pluginJarLocation, enabledLanguages, false, Optional.empty());
    var pluginLoader = new PluginsLoader().load(pluginConfiguration, Set.of());

    sonarlintEngine = new AnalysisEngine(sonarLintConfig, pluginLoader.getLoadedPlugins(), logOutput);
  }

  @AfterAll
  static void stop() {
    SonarLintLogger.setTarget(null);
    sonarlintEngine.stop();
  }

  @Test
  void shouldRaiseIssues() throws InterruptedException, ExecutionException {
    var inputFile = createInputFile(Path.of("projects/ipynb_project/file1.ipynb"), "file1.ipynb", false);
    var issues = new ArrayList<Issue>();

    var configuration = AnalysisConfiguration.builder()
      .setBaseDir(Path.of("projects/ipynb_project"))
      .addInputFile(inputFile)
      .addActiveRules(new ActiveRule("ipython:PrintStatementUsage", SonarLanguage.IPYTHON.name()),
        new ActiveRule("ipython:S1172", SonarLanguage.IPYTHON.name()),
        new ActiveRule("ipython:S930", SonarLanguage.IPYTHON.name()),
        new ActiveRule("ipython:S1542", SonarLanguage.IPYTHON.name()),
        new ActiveRule("ipython:BackticksUsage", SonarLanguage.IPYTHON.name()))
      .build();

    var logsByLevel = new HashMap<Level, List<String>>();
    var logOutput = createClientLogOutput(logsByLevel);
    var clientFileSystem = createClientFileSystem(inputFile);
    sonarlintEngine.post(new RegisterModuleCommand(new ClientModuleInfo("myModule", clientFileSystem)), progressMonitor).get();
    var command = new AnalyzeCommand("myModule", configuration, issues::add, logOutput);
    sonarlintEngine.post(command, progressMonitor).get();
    assertThat(issues)
      .extracting(Issue::getRuleKey, WithTextRange::getStartLine, i -> i.getInputFile().uri())
      .containsOnly(
        tuple("ipython:PrintStatementUsage", 32, inputFile.uri()),
        tuple("ipython:S1172", 40, inputFile.uri()),
        tuple("ipython:S930", 41, inputFile.uri()),
        tuple("ipython:S1172", 42, inputFile.uri()),
        tuple("ipython:S1542", 57, inputFile.uri()),
        tuple("ipython:BackticksUsage", 58, inputFile.uri()));
  }

  private static LogOutput createClientLogOutput(Map<Level, List<String>> logsByLevel) {
    return new LogOutput() {
      @Override
      public void log(String formattedMessage, Level level, @Nullable String stacktrace) {
        logsByLevel.computeIfAbsent(level, k -> new ArrayList<>()).add(formattedMessage);
      }
    };
  }

  private static ClientModuleFileSystem createClientFileSystem(ClientInputFile... inputFiles) {
    return new ClientModuleFileSystem() {
      @Override
      public Stream<ClientInputFile> files(String s, InputFile.Type type) {
        return Stream.of(inputFiles);
      }

      @Override
      public Stream<ClientInputFile> files() {
        return Stream.of(inputFiles);
      }
    };
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
