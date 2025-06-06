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
package org.sonar.plugins.python.flake8;

import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.plugins.python.Python;
import org.sonarsource.analyzer.commons.ExternalRuleLoader;

public class Flake8RulesDefinition implements RulesDefinition {

  private static final String RULES_JSON = "org/sonar/plugins/python/flake8/rules.json";

  private static final ExternalRuleLoader RULE_LOADER = new ExternalRuleLoader(Flake8Sensor.LINTER_KEY, Flake8Sensor.LINTER_NAME, RULES_JSON, Python.KEY);

  @Override
  public void define(RulesDefinition.Context context) {
    RULE_LOADER.createExternalRuleRepository(context);
  }

}
