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
package org.sonar.plugins.python.pylint;

import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.api.server.rule.RulesDefinitionXmlLoader;
import org.sonar.plugins.python.Python;

import static java.nio.charset.StandardCharsets.UTF_8;

public class PylintRuleRepository implements RulesDefinition {

  public static final String REPOSITORY_NAME = "Pylint";
  public static final String REPOSITORY_KEY = REPOSITORY_NAME;

  private static final String RULES_FILE = "/org/sonar/plugins/python/pylint/rules.xml";
  private static final String REMEDIATION_FILE = "/org/sonar/plugins/python/pylint/remediation-cost.csv";

  private final RulesDefinitionXmlLoader xmlLoader;

  public PylintRuleRepository(RulesDefinitionXmlLoader xmlLoader) {
    this.xmlLoader = xmlLoader;
  }

  @Override
  public void define(Context context) {
    NewRepository repository = context
      .createRepository(REPOSITORY_KEY, Python.KEY)
      .setName(REPOSITORY_NAME);
    xmlLoader.load(repository, getClass().getResourceAsStream(RULES_FILE), UTF_8.name());
    defineRemediationFunction(repository);
    repository.done();
  }

  private static void defineRemediationFunction(NewRepository repository) {
    Map<String, String> remediationCostMap = loadRemediationCostMap();
    for (NewRule rule : repository.rules()) {
      String gap = remediationCostMap.get(rule.key());
      if (gap == null) {
        throw new IllegalStateException("Missing remediation cost for rule " + rule.key());
      } else if (!gap.equals("null")) {
        rule.setDebtRemediationFunction(rule.debtRemediationFunctions().linear(gap));
      }
    }
  }

  private static Map<String, String> loadRemediationCostMap() {
    Map<String, String> map = new HashMap<>();
    try (Scanner scanner = new Scanner(PylintRuleRepository.class.getResourceAsStream(REMEDIATION_FILE), UTF_8.name())) {
      while (scanner.hasNext()) {
        String[] cols = scanner.next().split(",");
        map.put(cols[0], cols[1]);
      }
    }
    return map;
  }

}
