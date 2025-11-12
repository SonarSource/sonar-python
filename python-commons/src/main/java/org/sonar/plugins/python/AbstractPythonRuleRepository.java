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
package org.sonar.plugins.python;

import java.util.List;
import java.util.Set;
import org.sonar.api.SonarRuntime;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonarsource.analyzer.commons.RuleMetadataLoader;

public abstract class AbstractPythonRuleRepository implements RulesDefinition {
  private static final String REPOSITORY_NAME = "SonarAnalyzer";

  private final String repositoryKey;
  private final String resourceFolder;
  private final String languageKey;

  private final SonarRuntime sonarRuntime;

  protected AbstractPythonRuleRepository(String repositoryKey, String resourceFolder, String languageKey, SonarRuntime sonarRuntime) {
    this.repositoryKey = repositoryKey;
    this.resourceFolder = resourceFolder;
    this.languageKey = languageKey;
    this.sonarRuntime = sonarRuntime;
  }

  @Override
  public void define(Context context) {
    NewRepository repository = context
      .createRepository(repositoryKey, languageKey)
      .setName(REPOSITORY_NAME);


    RuleMetadataLoader loader = new RuleMetadataLoader(resourceFolder, resourceFolder + "/Sonar_way_profile.json", sonarRuntime);
    loader.addRulesByAnnotatedClass(repository, getCheckClasses());

    repository.rules().stream()
      .filter(rule -> getTemplateRuleKeys().contains(rule.key()))
      .forEach(rule -> rule.setTemplate(true));

    repository.rules().stream()
      .filter(rule -> getDisabledRules().contains(rule.key()))
      .forEach(rule -> rule.setActivatedByDefault(false));

    repository.done();
  }

  protected abstract List<Class<?>> getCheckClasses();

  protected Set<String> getTemplateRuleKeys() {
    return Set.of();
  }

  protected Set<String> getDisabledRules() {
    return Set.of();
  }
}
