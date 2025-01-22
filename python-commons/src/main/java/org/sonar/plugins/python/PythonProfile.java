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

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Collections;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.api.rule.RuleKey;
import org.sonar.api.server.profile.BuiltInQualityProfilesDefinition;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.python.checks.CheckList;
import org.sonarsource.analyzer.commons.BuiltInQualityProfileJsonLoader;

import static org.sonar.plugins.python.PythonRuleRepository.RESOURCE_FOLDER;

public class PythonProfile implements BuiltInQualityProfilesDefinition {

  private static final Logger LOG = LoggerFactory.getLogger(PythonProfile.class);

  static final String PROFILE_NAME = "Sonar way";
  static final String PROFILE_LOCATION = RESOURCE_FOLDER + "/Sonar_way_profile.json";
  static final String SECURITY_RULES_CLASS_NAME = "com.sonar.plugins.security.api.PythonRules";
  static final String SECURITY_RULE_KEYS_METHOD_NAME = "getRuleKeys";
  static final String DBD_RULES_CLASS_NAME = "com.sonarsource.plugins.dbd.api.PythonRules";
  static final String DBD_RULE_KEYS_METHOD_NAME = "getDataflowBugDetectionRuleKeys";
  static final String GET_REPOSITORY_KEY = "getRepositoryKey";


  @Override
  public void define(Context context) {
    NewBuiltInQualityProfile profile = context.createBuiltInQualityProfile(PROFILE_NAME, Python.KEY);
    BuiltInQualityProfileJsonLoader.load(profile, CheckList.REPOSITORY_KEY, PROFILE_LOCATION);
    getSecurityRuleKeys()
      .forEach(key -> profile.activateRule(key.repository(), key.rule()));
    getDataflowBugDetectionRuleKeys()
      .forEach(key -> profile.activateRule(key.repository(), key.rule()));
    profile.done();
  }

  static Set<RuleKey> getSecurityRuleKeys() {
    return getExternalRuleKeys(SECURITY_RULES_CLASS_NAME, SECURITY_RULE_KEYS_METHOD_NAME, "security");
  }

  static Set<RuleKey> getDataflowBugDetectionRuleKeys() {
    return getExternalRuleKeys(DBD_RULES_CLASS_NAME, DBD_RULE_KEYS_METHOD_NAME, "dataflow bug detection");
  }

  @SuppressWarnings("unchecked")
  static Set<RuleKey> getExternalRuleKeys(String className, String ruleKeysMethodName, String rulesCategory) {
    try {
      Class<?> rulesClass = Class.forName(className);
      Method getRuleKeysMethod = rulesClass.getMethod(ruleKeysMethodName);
      Set<String> ruleKeys = (Set<String>) getRuleKeysMethod.invoke(null);
      Method getRepositoryKeyMethod = rulesClass.getMethod(GET_REPOSITORY_KEY);
      String repositoryKey = (String) getRepositoryKeyMethod.invoke(null);
      return ruleKeys.stream().map(k -> RuleKey.of(repositoryKey, k)).collect(Collectors.toSet());
    } catch (ClassNotFoundException | NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
      LOG.debug(String.format("[%s], no %s rules added to Sonar way Python profile: %s", e.getClass().getSimpleName(), rulesCategory, e.getMessage()));
    }
    return Collections.emptySet();
  }
}
