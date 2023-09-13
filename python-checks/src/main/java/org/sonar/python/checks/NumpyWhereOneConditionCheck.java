package org.sonar.python.checks;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;

@Rule(key = "S6729")
public class NumpyWhereOneConditionCheck extends PythonSubscriptionCheck {
  @Override
  public void initialize(Context context) {
    // To be implemented.
  }
}
