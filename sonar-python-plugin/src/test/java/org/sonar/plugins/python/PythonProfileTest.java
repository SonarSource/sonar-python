/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import com.sonar.plugins.security.api.PythonRules;
import org.junit.jupiter.api.Test;
import org.sonar.api.rule.RuleKey;
import org.sonar.api.server.profile.BuiltInQualityProfilesDefinition;
import org.sonar.api.server.profile.BuiltInQualityProfilesDefinition.BuiltInActiveRule;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.plugins.python.PythonProfile.SECURITY_RULES_CLASS_NAME;
import static org.sonar.plugins.python.PythonProfile.SECURITY_RULE_KEYS_METHOD_NAME;
import static org.sonar.plugins.python.PythonProfile.GET_REPOSITORY_KEY;
import static org.sonar.plugins.python.PythonProfile.getDataflowBugDetectionRuleKeys;
import static org.sonar.plugins.python.PythonProfile.getExternalRuleKeys;
import static org.sonar.plugins.python.PythonProfile.getSecurityRuleKeys;

class PythonProfileTest {


  public BuiltInQualityProfilesDefinition.BuiltInQualityProfile getProfile() {
    BuiltInQualityProfilesDefinition.Context context = new BuiltInQualityProfilesDefinition.Context();
    new PythonProfile().define(context);
    return context.profile("py", "Sonar way");
  }

  @Test
  void profile() {
    BuiltInQualityProfilesDefinition.BuiltInQualityProfile profile = getProfile();
    assertThat(profile.rules()).extracting("repoKey").containsOnly("python", "pythonsecurity");
    assertThat(profile.rules().size()).isGreaterThan(25);
    assertThat(profile.rules()).extracting(BuiltInActiveRule::ruleKey).contains("S100");
  }

  @Test
  void should_contains_security_rules_if_available() {
    // no security rule available
    PythonRules.getRuleKeys().clear();
    assertThat(getSecurityRuleKeys())
      .isEmpty();

    // one security rule available
    PythonRules.getRuleKeys().add("S3649");
    assertThat(getSecurityRuleKeys())
      .containsOnly(RuleKey.of("pythonsecurity", "S3649"));

    PythonRules.throwOnCall = true;
    assertThat(getSecurityRuleKeys())
      .isEmpty();
    PythonRules.throwOnCall = false;
  }

  @Test
  void should_contains_dataflow_bug_detection_rules_if_available() {
    // no dataflow bug detection rules available
    com.sonarsource.plugins.dbd.api.PythonRules.getDataflowBugDetectionRuleKeys().clear();
    assertThat(getDataflowBugDetectionRuleKeys()).isEmpty();

    // one dataflow bug detection rule available
    com.sonarsource.plugins.dbd.api.PythonRules.getDataflowBugDetectionRuleKeys().add("S2259");
    assertThat(getDataflowBugDetectionRuleKeys()).containsOnly(RuleKey.of("dbd-repo-key", "S2259"));

    com.sonarsource.plugins.dbd.api.PythonRules.throwOnCall = true;
    assertThat(getDataflowBugDetectionRuleKeys())
      .isEmpty();
    PythonRules.throwOnCall = false;
  }

  @Test
  void test_get_external_rule_keys() {
    // invalid class name
    assertThat(getExternalRuleKeys("xxx", SECURITY_RULE_KEYS_METHOD_NAME, GET_REPOSITORY_KEY)).isEmpty();

    // invalid method name
    assertThat(getExternalRuleKeys(SECURITY_RULES_CLASS_NAME, "xxx", GET_REPOSITORY_KEY)).isEmpty();
  }
}
