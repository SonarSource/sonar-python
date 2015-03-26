package org.sonar.python.parser.python_v3;

import org.junit.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;

import static org.sonar.sslr.tests.Assertions.assertThat;

public class PythonV3Test extends RuleTest {

  @Test
  public void ellipsis(){
    setRootRule(PythonGrammar.TEST);
    assertThat(p).matches("...");
    assertThat(p).matches("x[...]");
  }

}
