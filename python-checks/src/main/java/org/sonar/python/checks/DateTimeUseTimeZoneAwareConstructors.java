package org.sonar.python.checks;

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S6903")
public class DateTimeUseTimeZoneAwareConstructors extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Using timezone aware \"datetime\"s should be preferred over using \"datetime.datetime.utcnow\" and \"datetime.datetime.utcfromtimestamp\"";

  private static final Set<String> NON_COMPLIANT_FQNS = Set.of("datetime.datetime.utcnow", "datetime.datetime.utcfromtimestamp");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCallExpr);
  }

  private void checkCallExpr(SubscriptionContext context) {
    CallExpression callExpression = (CallExpression) context.syntaxNode();
    Symbol calleeSymbol = callExpression.calleeSymbol();

    if (calleeSymbol != null && NON_COMPLIANT_FQNS.contains(calleeSymbol.fullyQualifiedName())) {
      var issue = context.addIssue(callExpression, MESSAGE);
      addQuickFix(context, issue);
    }
  }

}
