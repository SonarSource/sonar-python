package org.sonar.python.checks;

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class NumpyWhereOneConditionCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("/src/test/resources/checks/numpyWhereOneCondition.py", new NumpyWhereOneConditionCheck());
  }

}
