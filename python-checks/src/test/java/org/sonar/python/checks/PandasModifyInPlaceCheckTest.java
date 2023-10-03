package org.sonar.python.checks;

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class PandasModifyInPlaceCheckTest {
  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/pandasModifyInPlace.py", new PandasModifyInPlaceCheck());
  }
}
