package org.sonar.python.checks;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;

@Rule(key = "S6714")
public class NumpyListOverGeneratorCheck extends PythonSubscriptionCheck {
  @Override
  public void initialize(Context context) {
    // To be implemented.
  }
}
