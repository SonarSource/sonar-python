/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Collections;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.api.rule.RuleKey;
import org.sonar.api.server.profile.BuiltInQualityProfilesDefinition;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.python.checks.CheckList;
import org.sonarsource.analyzer.commons.BuiltInQualityProfileJsonLoader;

import static org.sonar.plugins.python.PythonRuleRepository.RESOURCE_FOLDER;

public class PythonProfile implements BuiltInQualityProfilesDefinition {

  private static final Logger LOG = Loggers.get(PythonProfile.class);

  static final String PROFILE_NAME = "Sonar way";
  static final String PROFILE_LOCATION = RESOURCE_FOLDER + "/Sonar_way_profile.json";
  static final String SECURITY_RULES_CLASS_NAME = "com.sonar.plugins.security.api.PythonRules";
  static final String SECURITY_RULE_KEYS_METHOD_NAME = "getRuleKeys";
  static final String SECURITY_RULE_REPO_METHOD_NAME = "getRepositoryKey";

  @Override
  public void define(Context context) {
    NewBuiltInQualityProfile profile = context.createBuiltInQualityProfile(PROFILE_NAME, Python.KEY);
    BuiltInQualityProfileJsonLoader.load(profile, CheckList.REPOSITORY_KEY, PROFILE_LOCATION);
    getSecurityRuleKeys(SECURITY_RULES_CLASS_NAME, SECURITY_RULE_KEYS_METHOD_NAME, SECURITY_RULE_REPO_METHOD_NAME)
      .forEach(key -> profile.activateRule(key.repository(), key.rule()));
    profile.done();
  }

  // Visible for testing
  static Set<RuleKey> getSecurityRuleKeys(String className, String ruleKeysMethodName, String ruleRepoMethodName) {
    try {

      Class<?> rulesClass = Class.forName(className);
      Method getRuleKeysMethod = rulesClass.getMethod(ruleKeysMethodName);
      Set<String> ruleKeys = (Set<String>) getRuleKeysMethod.invoke(null);
      Method getRepositoryKeyMethod = rulesClass.getMethod(ruleRepoMethodName);
      String repositoryKey = (String) getRepositoryKeyMethod.invoke(null);
      return ruleKeys.stream().map(k -> RuleKey.of(repositoryKey, k)).collect(Collectors.toSet());

    } catch (ClassNotFoundException e) {
      LOG.debug(className + " is not found, " + securityRuleMessage(e));
    } catch (NoSuchMethodException e) {
      LOG.debug("Method not found on " + className +", " + securityRuleMessage(e));
    } catch (IllegalAccessException | InvocationTargetException e) {
      LOG.debug(e.getClass().getSimpleName() + ": " + securityRuleMessage(e));
    }

    return Collections.emptySet();
  }

  private static String securityRuleMessage(Exception e) {
    return "no security rules added to Sonar way Python profile: " + e.getMessage();
  }
}
