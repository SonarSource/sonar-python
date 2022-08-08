package org.sonar.python.checks.tests;

import org.junit.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

public class TestShouldBeSkippedExplicitlyCheckTest {

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/testShouldBeSkippedExplicitly.py", new TestShouldBeSkippedExplicitlyCheck());
  }
}
