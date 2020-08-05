package org.sonar.python.checks;

import org.junit.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

public class FunctionReturnTypeCheckTest {

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/functionReturnType.py", new FunctionReturnTypeCheck());
  }
}
