package org.sonar.python.checks;

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class NumpyListOverGeneratorCheckTest {
  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/numpyListOverGenerator.py", new NumpyListOverGeneratorCheck());
  }
}
