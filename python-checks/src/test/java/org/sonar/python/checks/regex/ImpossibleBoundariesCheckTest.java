package org.sonar.python.checks.regex;

import org.junit.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;


public class ImpossibleBoundariesCheckTest {

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/regex/impossibleBoundariesCheck.py", new ImpossibleBoundariesCheck());
  }

}
