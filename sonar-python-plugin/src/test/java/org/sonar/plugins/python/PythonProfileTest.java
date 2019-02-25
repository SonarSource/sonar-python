/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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

import org.junit.Test;
import org.sonar.api.SonarRuntime;
import org.sonar.api.server.profile.BuiltInQualityProfilesDefinition;
import org.sonar.api.server.profile.BuiltInQualityProfilesDefinition.BuiltInActiveRule;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonProfileTest {


  public BuiltInQualityProfilesDefinition.BuiltInQualityProfile getProfile(SonarRuntime sonarRuntime) {
    BuiltInQualityProfilesDefinition.Context context = new BuiltInQualityProfilesDefinition.Context();
    new PythonProfile(sonarRuntime).define(context);
    return context.profile("py", "Sonar way");
  }

  @Test
  public void profile() {
    BuiltInQualityProfilesDefinition.BuiltInQualityProfile profile = getProfile(TestUtils.SONAR_RUNTIME_72);
    assertThat(profile.rules()).extracting("repoKey").containsOnly("python");
    assertThat(profile.rules().size()).isGreaterThan(25);
    assertThat(profile.rules()).extracting(BuiltInActiveRule::ruleKey).contains("S100");
  }

  @Test
  public void remove_hotspot_when_not_supported() {
    BuiltInQualityProfilesDefinition.BuiltInQualityProfile profile = getProfile(TestUtils.SONAR_RUNTIME_67);
    assertThat(profile.rules()).extracting(BuiltInActiveRule::ruleKey).doesNotContain("S1313");

    profile = getProfile(TestUtils.SONAR_RUNTIME_72);
    assertThat(profile.rules()).extracting(BuiltInActiveRule::ruleKey).contains("S1313");
  }

}
