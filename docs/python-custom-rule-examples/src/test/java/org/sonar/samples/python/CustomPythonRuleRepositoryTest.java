/*
 * Copyright (C) 2011-2024 SonarSource SA - mailto:info AT sonarsource DOT com
 * This code is released under [MIT No Attribution](https://opensource.org/licenses/MIT-0) license.
 */
package org.sonar.samples.python;

import java.io.IOException;
import java.net.URL;
import org.junit.Test;
import org.sonar.api.server.rule.RulesDefinition;
import org.mockito.Mockito;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class CustomPythonRuleRepositoryTest {

  @Test
  public void test_rule_repository() {
    CustomPythonRuleRepository customPythonRuleRepository = new CustomPythonRuleRepository();
    RulesDefinition.Context context = new RulesDefinition.Context();
    customPythonRuleRepository.define(context);
    assertThat(customPythonRuleRepository.repositoryKey()).isEqualTo("python-custom-rules");
    assertThat(context.repositories()).hasSize(1).extracting("key").containsExactly(customPythonRuleRepository.repositoryKey());
    assertThat(context.repositories().get(0).rules()).hasSize(2);
    assertThat(customPythonRuleRepository.checkClasses()).hasSize(2);
  }

  @Test
  public void test_unfound_resource(){
    assertThatThrownBy(() -> new CustomPythonRuleRepository().loadResource("/unknown"))
      .isInstanceOf(IllegalStateException.class)
      .hasMessage("Resource not found: /unknown");
  }

  @Test
  public void test_read_exception_resource() throws IOException {
    URL urlMock = Mockito.mock(URL.class);
    Mockito.when(urlMock.openStream()).thenThrow(IOException.class);
    Mockito.when(urlMock.toString()).thenReturn("MyURL");
    assertThatThrownBy(() -> CustomPythonRuleRepository.readResource(urlMock))
      .isInstanceOf(IllegalStateException.class)
      .hasMessage("Failed to read resource: MyURL");
  }
}
