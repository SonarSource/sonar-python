/*
 * Copyright (C) SonarSource Sàrl - mailto:info AT sonarsource DOT com
 * This code is released under [MIT No Attribution](https://opensource.org/licenses/MIT-0) license.
 */
package org.sonar.samples.python;

import com.sonarsource.scanner.engine.sensor.test.fixtures.TestSonarRuntime;
import org.junit.jupiter.api.Test;
import org.sonar.api.Plugin;
import org.sonar.api.SonarEdition;
import org.sonar.api.SonarQubeSide;
import org.sonar.api.SonarRuntime;
import org.sonar.api.utils.Version;

import static org.assertj.core.api.Assertions.assertThat;

class CustomPythonRulesPluginTest {
  @Test
  void test() {
    SonarRuntime sonarRuntime = TestSonarRuntime.forSonarQube(Version.create(9, 9), SonarQubeSide.SCANNER, SonarEdition.DEVELOPER);
    Plugin.Context context = new Plugin.Context(sonarRuntime);
    new CustomPythonRulesPlugin().define(context);
    assertThat(context.getExtensions()).hasSize(1);
  }
}
