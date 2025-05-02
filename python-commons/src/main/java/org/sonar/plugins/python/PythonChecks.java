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
package org.sonar.plugins.python;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.api.batch.rule.CheckFactory;
import org.sonar.api.batch.rule.Checks;
import org.sonar.api.rule.RuleKey;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonCustomRuleRepository;
import org.sonar.plugins.python.api.internal.EndOfAnalysis;

public class PythonChecks {
  private static final Set<String> SONAR_PYTHON_REPOSITORIES = Set.of("python", "pythonenterprise", "ipython", "ipythonenterprise");

  private final CheckFactory checkFactory;
  private final Map<String, RepositoryChecksInfo> sonarPythonRepositoriesChecks;
  private final Map<String, Checks<PythonCheck>> noSonarPythonRepositoriesChecks;
  private final Map<Class<? extends PythonCheck>, RuleKey> ruleKeys;

  PythonChecks(CheckFactory checkFactory) {
    this.checkFactory = checkFactory;
    this.sonarPythonRepositoriesChecks = new ConcurrentHashMap<>();
    this.noSonarPythonRepositoriesChecks = new ConcurrentHashMap<>();
    this.ruleKeys = new ConcurrentHashMap<>();
  }

  public PythonChecks addChecks(String repositoryKey, Iterable<Class<?>> checkClasses) {
    var repositoryChecksInfo = new RepositoryChecksInfo(repositoryKey, checkClasses);
    var checks = createChecks(repositoryChecksInfo);
    checks.all().forEach(check -> {
      var checkClass = check.getClass();
      var ruleKey = checks.ruleKey(check);
      ruleKeys.put(checkClass, ruleKey);
    });
    if (SONAR_PYTHON_REPOSITORIES.contains(repositoryKey)) {
      sonarPythonRepositoriesChecks.put(repositoryChecksInfo.repositoryKey, repositoryChecksInfo);
    } else {
      noSonarPythonRepositoriesChecks.put(repositoryChecksInfo.repositoryKey, createChecks(repositoryChecksInfo));
    }
    return this;
  }

  public PythonChecks addCustomChecks(@Nullable PythonCustomRuleRepository[] customRuleRepositories) {
    Stream.ofNullable(customRuleRepositories)
      .flatMap(Stream::of)
      .forEach(ruleRepository -> addChecks(ruleRepository.repositoryKey(), ruleRepository.checkClasses()));
    return this;
  }

  public synchronized List<PythonCheck> sonarPythonChecks() {
    return sonarPythonRepositoriesChecks.values().stream()
      .map(this::createChecks)
      .map(Checks::all)
      .flatMap(Collection::stream)
      .toList();
  }

  public synchronized Map<String, List<PythonCheck>> noSonarPythonChecks() {
    return noSonarPythonRepositoriesChecks.entrySet()
      .stream()
      .collect(Collectors.toMap(Map.Entry::getKey,
        entry -> entry.getValue().all().stream().toList())
      );
  }

  public List<EndOfAnalysis> sonarPythonEndOfAnalyses() {
    return sonarPythonChecks().stream()
      .filter(EndOfAnalysis.class::isInstance)
      .map(EndOfAnalysis.class::cast)
      .toList();
  }

  public Map<String, List<EndOfAnalysis>> noSonarPythonEndOfAnalyses() {
    return noSonarPythonChecks().entrySet().stream()
      .collect(Collectors.toMap(Map.Entry::getKey, entry -> entry.getValue().stream()
        .filter(EndOfAnalysis.class::isInstance)
        .map(EndOfAnalysis.class::cast)
        .toList()));
  }

  @Nullable
  public RuleKey ruleKey(PythonCheck check) {
    return ruleKeys.getOrDefault(check.getClass(), null);
  }

  private Checks<PythonCheck> createChecks(RepositoryChecksInfo repositoryChecksInfo) {
    return checkFactory.<PythonCheck>create(repositoryChecksInfo.repositoryKey).addAnnotatedChecks(repositoryChecksInfo.checkClasses);
  }

  private record RepositoryChecksInfo(String repositoryKey, Iterable<Class<?>> checkClasses) {}

}
