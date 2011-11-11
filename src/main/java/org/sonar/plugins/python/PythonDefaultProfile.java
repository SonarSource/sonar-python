/*
 * Sonar Python Plugin
 * Copyright (C) 2011 Waleri Enns
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

package org.sonar.plugins.python;

import org.sonar.api.profiles.ProfileDefinition;
import org.sonar.api.profiles.RulesProfile;
import org.sonar.api.rules.Rule;
import org.sonar.api.utils.ValidationMessages;

public class PythonDefaultProfile extends ProfileDefinition {

  private PythonRuleRepository repository;

  public PythonDefaultProfile(PythonRuleRepository repository) {
    this.repository = repository;
  }

  @Override
  public RulesProfile createProfile(ValidationMessages validation) {
    RulesProfile rulesProfile = RulesProfile.create("Default Python Profile", Python.KEY);

    for (Rule rule : repository.createRules()) {
      rulesProfile.activateRule(rule, null);
    }

    return rulesProfile;
  }
}
