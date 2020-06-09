package org.sonar.python.checks;

import org.junit.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

public class StringFormatCorrectnessCheckTest {

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/stringFormatCorrectness.py", new StringFormatCorrectnessCheck());
  }
}
