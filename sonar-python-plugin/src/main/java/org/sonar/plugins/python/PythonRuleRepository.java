/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
package org.sonar.plugins.python;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.python.checks.CheckList;
import org.sonarsource.analyzer.commons.RuleMetadataLoader;

public class PythonRuleRepository implements RulesDefinition {

  private static final String REPOSITORY_NAME = "SonarAnalyzer";

  private static final String RESOURCE_FOLDER = "org/sonar/l10n/py/rules/python";

  private static final Set<String> TEMPLATE_RULE_KEYS = new HashSet<>(Arrays.asList("XPath", "CommentRegularExpression"));

  @Override
  public void define(Context context) {
    NewRepository repository = context
      .createRepository(CheckList.REPOSITORY_KEY, Python.KEY)
      .setName(REPOSITORY_NAME);

    getRuleMetadataLoader().addRulesByAnnotatedClass(repository, getCheckClasses());

    repository.rules().stream()
      .filter(rule -> TEMPLATE_RULE_KEYS.contains(rule.key()))
      .forEach(rule -> rule.setTemplate(true));

    repository.done();
  }

  private static RuleMetadataLoader getRuleMetadataLoader() {
    return new RuleMetadataLoader(RESOURCE_FOLDER, PythonProfile.PROFILE_LOCATION);
  }

  private static List<Class> getCheckClasses() {
    return StreamSupport.stream(CheckList.getChecks().spliterator(), false)
      .collect(Collectors.toList());
  }

}
