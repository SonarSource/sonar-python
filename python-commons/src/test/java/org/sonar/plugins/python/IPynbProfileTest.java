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

import org.junit.jupiter.api.Test;
import org.sonar.api.server.profile.BuiltInQualityProfilesDefinition;

import static org.assertj.core.api.Assertions.assertThat;

class IPynbProfileTest {

  public BuiltInQualityProfilesDefinition.BuiltInQualityProfile getProfile() {
    BuiltInQualityProfilesDefinition.Context context = new BuiltInQualityProfilesDefinition.Context();
    new IPynbProfile().define(context);
    return context.profile("ipynb", "Sonar way");
  }

  @Test
  void profile() {
    BuiltInQualityProfilesDefinition.BuiltInQualityProfile profile = getProfile();
    assertThat(profile.rules()).extracting("repoKey").containsOnly("ipython");
    assertThat(profile.rules()).hasSizeGreaterThan(25);
    assertThat(profile.rules()).extracting(BuiltInQualityProfilesDefinition.BuiltInActiveRule::ruleKey).contains("S100");
    assertThat(profile.rules()).extracting(BuiltInQualityProfilesDefinition.BuiltInActiveRule::ruleKey).doesNotContain("S905");
    assertThat(profile.rules()).extracting(BuiltInQualityProfilesDefinition.BuiltInActiveRule::ruleKey).doesNotContain("S2201");
    assertThat(profile.rules()).extracting(BuiltInQualityProfilesDefinition.BuiltInActiveRule::ruleKey).doesNotContain("S5754");
    assertThat(profile.rules()).extracting(BuiltInQualityProfilesDefinition.BuiltInActiveRule::ruleKey).doesNotContain("S1481");
  }
}
