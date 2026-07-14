/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import static org.awaitility.Awaitility.await;

import com.sonar.python.it.PluginLocator;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.regex.Pattern;
import org.sonarsource.sonarlint.core.rpc.protocol.backend.file.DidOpenFileParams;
import org.sonarsource.sonarlint.core.rpc.protocol.backend.rules.StandaloneRuleConfigDto;
import org.sonarsource.sonarlint.core.rpc.protocol.backend.rules.UpdateStandaloneRulesConfigurationParams;
import org.sonarsource.sonarlint.core.rpc.protocol.client.issue.RaisedIssueDto;
import org.sonarsource.sonarlint.core.rpc.protocol.common.ClientFileDto;
import org.sonarsource.sonarlint.core.rpc.protocol.common.Language;
import org.sonarsource.sonarlint.core.test.utils.SonarLintBackendFixture;
import org.sonarsource.sonarlint.core.test.utils.SonarLintTestRpcServer;
import org.sonarsource.sonarlint.core.test.utils.junit5.SonarLintTestHarness;
import org.sonarsource.sonarlint.core.test.utils.plugins.Plugin;

final class SonarLintBackendTestUtils {

  static final String CONFIG_SCOPE_ID = "python-sonarlint-test";
  private static final Duration TIMEOUT = Duration.ofSeconds(15);

  private SonarLintBackendTestUtils() {
    // utility class
  }

  static ClientFileDto writeFile(Path baseDir, String relativePath, String content, Language language) {
    var filePath = baseDir.resolve(relativePath);
    try {
      Files.createDirectories(filePath.getParent());
      Files.writeString(filePath, content, StandardCharsets.UTF_8);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    return clientFile(baseDir, filePath, language);
  }

  static ClientFileDto copyFile(Path baseDir, Path sourceFile, String relativePath, Language language) {
    var targetPath = baseDir.resolve(relativePath);
    try {
      Files.createDirectories(targetPath.getParent());
      Files.copy(sourceFile, targetPath);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    return clientFile(baseDir, targetPath, language);
  }

  static SonarLintBackendFixture.FakeSonarLintRpcClient createClient(SonarLintTestHarness harness, Path baseDir, ClientFileDto... inputFiles) {
    return harness.newFakeClient()
      .withInitialFs(CONFIG_SCOPE_ID, baseDir, List.of(inputFiles))
      .build();
  }

  static SonarLintTestRpcServer startBackend(SonarLintTestHarness harness, SonarLintBackendFixture.FakeSonarLintRpcClient client,
    Set<Language> languages, Set<String> activeRuleKeys) {
    var backend = harness.newBackend()
      .withStandaloneEmbeddedPluginAndEnabledLanguage(new Plugin(languages, pythonPluginLocation(), "", ""))
      .withUnboundConfigScope(CONFIG_SCOPE_ID)
      .start(client);

    var rulesByKey = backend.getRulesService().listAllStandaloneRulesDefinitions().join().getRulesByKey();
    if (!rulesByKey.keySet().containsAll(activeRuleKeys)) {
      throw new IllegalStateException("Missing standalone SonarLint rules: " + activeRuleKeys.stream()
        .filter(ruleKey -> !rulesByKey.containsKey(ruleKey))
        .toList());
    }

    var ruleConfigByKey = rulesByKey.values().stream()
      .filter(rule -> languages.contains(rule.getLanguage()))
      .collect(Collectors.toMap(
        rule -> rule.getKey(),
        rule -> new StandaloneRuleConfigDto(activeRuleKeys.contains(rule.getKey()), Map.of())));
    backend.getRulesService().updateStandaloneRulesConfiguration(new UpdateStandaloneRulesConfigurationParams(ruleConfigByKey));
    return backend;
  }

  static void openFile(SonarLintTestRpcServer backend, ClientFileDto inputFile) {
    backend.getFileService().didOpenFile(new DidOpenFileParams(CONFIG_SCOPE_ID, inputFile.getUri()));
  }

  static void awaitIssues(SonarLintBackendFixture.FakeSonarLintRpcClient client,
    Consumer<Map<java.net.URI, List<RaisedIssueDto>>> assertion) {
    await()
      .atMost(TIMEOUT)
      .untilAsserted(() -> assertion.accept(client.getRaisedIssuesForScopeId(CONFIG_SCOPE_ID)));
  }

  private static ClientFileDto clientFile(Path baseDir, Path filePath, Language language) {
    var absoluteBaseDir = baseDir.toAbsolutePath().normalize();
    var absoluteFilePath = filePath.toAbsolutePath().normalize();
    return new ClientFileDto(
      absoluteFilePath.toUri(),
      absoluteBaseDir.relativize(absoluteFilePath),
      CONFIG_SCOPE_ID,
      false,
      StandardCharsets.UTF_8.name(),
      absoluteFilePath,
      null,
      language,
      true);
  }

  private static Path pythonPluginLocation() {
    return PluginLocator.isEnterpriseTest()
      ? findPluginArtifact(Path.of("../../../private/sonar-python-enterprise-plugin/target"),
        Pattern.compile("sonar-python-enterprise-plugin-[0-9.]*(?:-SNAPSHOT)?\\.jar"))
      : findPluginArtifact(Path.of("../../../sonar-python-plugin/target"),
        Pattern.compile("sonar-python-plugin-[0-9.]*(?:-SNAPSHOT)?\\.jar"));
  }

  private static Path findPluginArtifact(Path targetDir, Pattern filenamePattern) {
    var absoluteTargetDir = targetDir.toAbsolutePath().normalize();
    try (var entries = Files.list(absoluteTargetDir)) {
      return entries
        .filter(path -> filenamePattern.matcher(path.getFileName().toString()).matches())
        .findFirst()
        .orElseThrow(() -> new IllegalStateException("Cannot find plugin artifact in " + absoluteTargetDir));
    } catch (IOException e) {
      throw new UncheckedIOException("Failed to resolve plugin artifact from " + absoluteTargetDir, e);
    }
  }
}
