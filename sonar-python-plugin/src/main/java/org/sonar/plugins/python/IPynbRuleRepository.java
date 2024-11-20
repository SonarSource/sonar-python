/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;
import org.sonar.api.SonarRuntime;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.python.checks.CheckList;
import org.sonarsource.analyzer.commons.RuleMetadataLoader;

public class IPynbRuleRepository implements RulesDefinition {

  private static final String REPOSITORY_NAME = "SonarAnalyzer";

  static final String RESOURCE_FOLDER = "org/sonar/l10n/py/rules/python";

  private static final Set<String> TEMPLATE_RULE_KEYS = Collections.singleton("CommentRegularExpression");

  private final SonarRuntime runtime;

  public IPynbRuleRepository(SonarRuntime runtime) {
    this.runtime = runtime;
  }

  @Override
  public void define(Context context) {
    NewRepository repository = context
      .createRepository(CheckList.IPYTHON_REPOSITORY_KEY, IPynb.KEY)
      .setName(REPOSITORY_NAME);

    RuleMetadataLoader loader = new RuleMetadataLoader(RESOURCE_FOLDER, PythonProfile.PROFILE_LOCATION, runtime);
    loader.addRulesByAnnotatedClass(repository, getCheckClasses());

    repository.rules().stream()
      .filter(rule -> TEMPLATE_RULE_KEYS.contains(rule.key()))
      .forEach(rule -> rule.setTemplate(true));

    repository.rules().stream()
      .filter(rule -> IPynbProfile.DISABLED_RULES.contains(rule.key()))
      .forEach(rule -> rule.setActivatedByDefault(false));

    repository.done();
  }

  private static List<Class<?>> getCheckClasses() {
    return StreamSupport.stream(CheckList.getChecks().spliterator(), false)
      .map(check -> (Class<?>) check)
      .collect(Collectors.toList());
  }

}
