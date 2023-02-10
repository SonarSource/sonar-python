/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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

import org.sonar.api.server.profile.BuiltInQualityProfilesDefinition;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonarsource.analyzer.commons.BuiltInQualityProfileJsonLoader;

import static org.sonar.plugins.python.PythonRuleRepository.RESOURCE_FOLDER;

public class IPynbProfile implements BuiltInQualityProfilesDefinition {

  private static final Logger LOG = Loggers.get(IPynbProfile.class);

  static final String PROFILE_NAME = "Sonar way";
  static final String PROFILE_LOCATION = RESOURCE_FOLDER + "/Sonar_way_profile.json";
  static final String SECURITY_RULES_CLASS_NAME = "com.sonar.plugins.security.api.PythonRules";
  static final String SECURITY_RULE_KEYS_METHOD_NAME = "getRuleKeys";
  static final String DBD_RULES_CLASS_NAME = "com.sonarsource.plugins.dbd.api.PythonRules";
  static final String DBD_RULE_KEYS_METHOD_NAME = "getDataflowBugDetectionRuleKeys";
  static final String GET_REPOSITORY_KEY = "getRepositoryKey";


  @Override
  public void define(Context context) {
    NewBuiltInQualityProfile profile = context.createBuiltInQualityProfile(PROFILE_NAME, IPynb.KEY);
    BuiltInQualityProfileJsonLoader.load(profile, "ipython", PROFILE_LOCATION);
    profile.done();
  }
}
