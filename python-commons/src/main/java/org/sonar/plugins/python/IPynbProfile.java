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

import java.util.Set;
import org.sonar.api.server.profile.BuiltInQualityProfilesDefinition;
import org.sonar.python.checks.CheckList;
import org.sonarsource.analyzer.commons.BuiltInQualityProfileJsonLoader;

import static org.sonar.plugins.python.PythonRuleRepository.RESOURCE_FOLDER;

public class IPynbProfile implements BuiltInQualityProfilesDefinition {

  static final String PROFILE_NAME = "Sonar way";
  static final String PROFILE_LOCATION = RESOURCE_FOLDER + "/Sonar_way_profile.json";
  static final Set<String> DISABLED_RULES = Set.of("S905", "S2201", "S5754", "S1481");

  @Override
  public void define(Context context) {
    NewBuiltInQualityProfile profile = context.createBuiltInQualityProfile(PROFILE_NAME, IPynb.KEY);
    BuiltInQualityProfileJsonLoader.load(profile, CheckList.IPYTHON_REPOSITORY_KEY, PROFILE_LOCATION);
    profile.activeRules().removeIf(IPynbProfile::isDisabled);
    profile.done();
  }

  /**
   * Some rules from the default Python quality profile are considered noisy in IPython notebooks context
   * They are therefore filtered out of the default profile.
   */
  private static boolean isDisabled(NewBuiltInActiveRule rule) {
    return DISABLED_RULES.contains(rule.ruleKey());
  }
}
