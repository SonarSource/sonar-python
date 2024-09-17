package org.sonar.python.checks;

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

public class TorchModuleShouldCallInitCheckTest {
  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/torchModuleShouldCallInit.py", new TorchModuleShouldCallInitCheck());
  }
}
