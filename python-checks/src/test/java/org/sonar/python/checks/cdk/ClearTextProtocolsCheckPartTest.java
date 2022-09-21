package org.sonar.python.checks.cdk;

import org.junit.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

public class ClearTextProtocolsCheckPartTest {

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/cdk/clearTextProtocolsCheck.py", new ClearTextProtocolsCheckPart());
  }

}
