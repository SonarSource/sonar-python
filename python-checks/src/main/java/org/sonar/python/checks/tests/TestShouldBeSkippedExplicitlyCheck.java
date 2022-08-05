package org.sonar.python.checks.tests;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S5918")
public class TestShouldBeSkippedExplicitlyCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.RETURN_STMT, ctx -> {
      ReturnStatement returnStatement = (ReturnStatement) ctx.syntaxNode();

      // ensure that the return statment is conditional
      returnStatement.parent()
    });
  }
}
