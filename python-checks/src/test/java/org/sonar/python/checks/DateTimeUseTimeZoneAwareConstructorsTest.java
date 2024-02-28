package org.sonar.python.checks;

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class DateTimeUseTimeZoneAwareConstructorsTest {
  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/datetime_constructor_use_timezone_aware.py", new DateTimeUseTimeZoneAwareConstructors());
  }
}