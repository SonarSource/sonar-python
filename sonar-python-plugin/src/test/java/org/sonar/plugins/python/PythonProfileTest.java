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

import com.sonar.plugins.security.api.PythonRules;
import org.junit.Test;
import org.sonar.api.rule.RuleKey;
import org.sonar.api.server.profile.BuiltInQualityProfilesDefinition;
import org.sonar.api.server.profile.BuiltInQualityProfilesDefinition.BuiltInActiveRule;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.plugins.python.PythonProfile.SECURITY_RULES_CLASS_NAME;
import static org.sonar.plugins.python.PythonProfile.SECURITY_RULE_KEYS_METHOD_NAME;
import static org.sonar.plugins.python.PythonProfile.SECURITY_RULE_REPO_METHOD_NAME;
import static org.sonar.plugins.python.PythonProfile.getSecurityRuleKeys;

public class PythonProfileTest {


  public BuiltInQualityProfilesDefinition.BuiltInQualityProfile getProfile() {
    BuiltInQualityProfilesDefinition.Context context = new BuiltInQualityProfilesDefinition.Context();
    new PythonProfile().define(context);
    return context.profile("py", "Sonar way");
  }

  @Test
  public void profile() {
    BuiltInQualityProfilesDefinition.BuiltInQualityProfile profile = getProfile();
    assertThat(profile.rules()).extracting("repoKey").containsOnly("python", "pythonsecurity");
    assertThat(profile.rules().size()).isGreaterThan(25);
    assertThat(profile.rules()).extracting(BuiltInActiveRule::ruleKey).contains("S100");
  }

  @Test
  public void should_contains_security_rules_if_available() {
    // no security rule available
    PythonRules.getRuleKeys().clear();
    assertThat(getSecurityRuleKeys(SECURITY_RULES_CLASS_NAME, SECURITY_RULE_KEYS_METHOD_NAME, SECURITY_RULE_REPO_METHOD_NAME))
      .isEmpty();

    // one security rule available
    PythonRules.getRuleKeys().add("S3649");
    assertThat(getSecurityRuleKeys(SECURITY_RULES_CLASS_NAME, SECURITY_RULE_KEYS_METHOD_NAME, SECURITY_RULE_REPO_METHOD_NAME))
      .containsOnly(RuleKey.of("pythonsecurity", "S3649"));

    // invalid class name
    assertThat(getSecurityRuleKeys("xxx", SECURITY_RULE_KEYS_METHOD_NAME, SECURITY_RULE_REPO_METHOD_NAME)).isEmpty();

    // invalid method name
    assertThat(getSecurityRuleKeys(SECURITY_RULES_CLASS_NAME, "xxx", SECURITY_RULE_REPO_METHOD_NAME)).isEmpty();

    PythonRules.throwOnCall = true;
    assertThat(getSecurityRuleKeys(SECURITY_RULES_CLASS_NAME, SECURITY_RULE_KEYS_METHOD_NAME, SECURITY_RULE_REPO_METHOD_NAME))
      .isEmpty();
    PythonRules.throwOnCall = false;
  }
}
