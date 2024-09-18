package org.sonar.python.checks;

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class TorchModuleModeShouldBeSetAfterLoadingCheckTest {
  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/torchModuleModeShouldBeSetAfterLoadingCheck.py", new TorchModuleModeShouldBeSetAfterLoadingCheck());
  }
}