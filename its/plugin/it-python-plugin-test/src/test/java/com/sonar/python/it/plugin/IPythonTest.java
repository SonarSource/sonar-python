/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2023 SonarSource SA
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
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.ClassRule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.sonar.api.batch.fs.InputFile;
import org.sonarsource.sonarlint.core.StandaloneSonarLintEngineImpl;
import org.sonarsource.sonarlint.core.analysis.api.ClientInputFile;
import org.sonarsource.sonarlint.core.analysis.api.ClientModuleFileSystem;
import org.sonarsource.sonarlint.core.analysis.api.ClientModuleInfo;
import org.sonarsource.sonarlint.core.analysis.api.WithTextRange;
import org.sonarsource.sonarlint.core.client.api.common.analysis.Issue;
import org.sonarsource.sonarlint.core.client.api.standalone.StandaloneAnalysisConfiguration;
import org.sonarsource.sonarlint.core.client.api.standalone.StandaloneGlobalConfiguration;
import org.sonarsource.sonarlint.core.client.api.standalone.StandaloneSonarLintEngine;
import org.sonarsource.sonarlint.core.commons.IssueSeverity;
import org.sonarsource.sonarlint.core.commons.Language;
import org.sonarsource.sonarlint.core.commons.log.ClientLogOutput;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.tuple;

public class IPythonTest {

  @ClassRule
  public static final TemporaryFolder TEMP = new TemporaryFolder();

  private static StandaloneSonarLintEngine sonarlintEngine;

  @BeforeClass
  public static void prepare() throws Exception {
    StandaloneGlobalConfiguration sonarLintConfig = StandaloneGlobalConfiguration.builder()
      .addPlugin(Tests.PLUGIN_LOCATION.getFile().toPath())
      .setSonarLintUserHome(TEMP.newFolder().toPath())
      .addEnabledLanguage(Language.IPYTHON)
      .setLogOutput((formattedMessage, level) -> {
        /* Don't pollute logs */ })
      .setModulesProvider(Collections::emptyList)
      .build();
    sonarlintEngine = new StandaloneSonarLintEngineImpl(sonarLintConfig);
  }

  @AfterClass
  public static void stop() {
    sonarlintEngine.stop();
  }

  @Test
  public void shouldRaiseIssues() {
    var inputFile = createInputFile(Path.of("projects/ipynb_project/file1.ipynb"), "file1.ipynb", false);
    var issues = new ArrayList<Issue>();

    var configuration = StandaloneAnalysisConfiguration.builder()
      .setBaseDir(Path.of("projects/ipynb_project"))
      .addInputFile(inputFile)
      .setModuleKey("myModule")
      .build();

    var logsByLevel = new HashMap<ClientLogOutput.Level, List<String>>();
    var logOutput = createClientLogOutput(logsByLevel);
    var clientFileSystem = createClientFileSystem(inputFile);
    sonarlintEngine.declareModule(new ClientModuleInfo("myModule", clientFileSystem));
    sonarlintEngine.analyze(configuration, issues::add, logOutput, null);

    assertThat(issues)
      .extracting(Issue::getRuleKey, WithTextRange::getStartLine, i -> i.getInputFile().uri(), Issue::getSeverity)
      .containsOnly(
        tuple("ipython:PrintStatementUsage", 32, inputFile.uri(), IssueSeverity.MAJOR),
        tuple("ipython:S1172", 40, inputFile.uri(), IssueSeverity.MAJOR),
        tuple("ipython:S930", 41, inputFile.uri(), IssueSeverity.BLOCKER),
        tuple("ipython:S1172", 42, inputFile.uri(), IssueSeverity.MAJOR),
        tuple("ipython:S1542", 57, inputFile.uri(), IssueSeverity.MAJOR),
        tuple("ipython:BackticksUsage", 58, inputFile.uri(), IssueSeverity.BLOCKER)
      );
  }

  private static ClientLogOutput createClientLogOutput(Map<ClientLogOutput.Level, List<String>> logsByLevel) {
    return (s, level) -> logsByLevel.computeIfAbsent(level, (k) -> new ArrayList<>()).add(s);
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
