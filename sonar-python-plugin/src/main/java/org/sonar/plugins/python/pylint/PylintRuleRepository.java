/*
 * SonarQube Python Plugin
 * Copyright (C) 2011 SonarSource and Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */
package org.sonar.plugins.python.pylint;

import org.sonar.api.rules.Rule;
import org.sonar.api.rules.RuleRepository;
import org.sonar.api.rules.XMLRuleParser;
import org.sonar.plugins.python.Python;

import java.util.List;

public class PylintRuleRepository extends RuleRepository {

  public static final String REPOSITORY_NAME = "Pylint";
  public static final String REPOSITORY_KEY = REPOSITORY_NAME;

  private static final String RULES_FILE = "/org/sonar/plugins/python/pylint/rules.xml";
  private final XMLRuleParser ruleParser;

  public PylintRuleRepository(XMLRuleParser ruleParser) {
    super(REPOSITORY_KEY, Python.KEY);
    setName(REPOSITORY_NAME);
    this.ruleParser = ruleParser;
  }

  @Override
  public List<Rule> createRules() {
    List<Rule> rules = ruleParser.parse(getClass().getResourceAsStream(RULES_FILE));
    for (Rule r : rules) {
      r.setRepositoryKey(REPOSITORY_KEY);
    }
    return rules;
  }

}
