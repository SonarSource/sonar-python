/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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

import com.google.common.base.Charsets;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.api.server.rule.RulesDefinitionXmlLoader;
import org.sonar.plugins.python.Python;
import org.sonar.squidbridge.rules.SqaleXmlLoader;

public class PylintRuleRepository implements RulesDefinition {

  public static final String REPOSITORY_NAME = "Pylint";
  public static final String REPOSITORY_KEY = REPOSITORY_NAME;

  private static final String RULES_FILE = "/org/sonar/plugins/python/pylint/rules.xml";
  private static final String SQALE_FILE = "/com/sonar/sqale/python-model.xml";
  private final RulesDefinitionXmlLoader xmlLoader;

  public PylintRuleRepository(RulesDefinitionXmlLoader xmlLoader) {
    this.xmlLoader = xmlLoader;
  }

  @Override
  public void define(Context context) {
    NewRepository repository = context
        .createRepository(REPOSITORY_KEY, Python.KEY)
        .setName(REPOSITORY_NAME);
    xmlLoader.load(repository, getClass().getResourceAsStream(RULES_FILE), Charsets.UTF_8.name());
    SqaleXmlLoader.load(repository, SQALE_FILE);
    repository.done();
  }
}
