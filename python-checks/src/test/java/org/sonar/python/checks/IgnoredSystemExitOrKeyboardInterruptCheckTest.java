package org.sonar.python.checks;

import org.junit.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

public class IgnoredSystemExitOrKeyboardInterruptCheckTest {

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/ignoredSystemExitOrKeyboardInterrupt.py", new IgnoredSystemExitOrKeyboardInterruptCheck());
  }

}
