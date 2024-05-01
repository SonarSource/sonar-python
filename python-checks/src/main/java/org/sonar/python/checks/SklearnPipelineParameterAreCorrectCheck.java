package org.sonar.python.checks;

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S6972")
public class SklearnPipelineParameterAreCorrectCheck  extends PythonSubscriptionCheck {
  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, SklearnPipelineParameterAreCorrectCheck::checkCallExpression);
  }

  private static void checkCallExpression(SubscriptionContext subscriptionContext) {
    CallExpression callExpression = (CallExpression) subscriptionContext.syntaxNode();

    var a = Optional.of(callExpression).map(CallExpression::calleeSymbol).
    map(Symbol::fullyQualifiedName)
      .filter("sklearn.pipeline.Pipeline.set_params"::equals);

    if (!a.isPresent()) {
      return;
    }

    System.out.println(a);
  }
}
