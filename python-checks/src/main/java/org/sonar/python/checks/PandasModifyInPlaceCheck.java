package org.sonar.python.checks;

import java.util.Optional;
import java.util.Set;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

public class PandasModifyInPlaceCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Do not use 'inplace=True' when modifying a dataframe.";

  private static final Set<String> FULLY_QUALIFIED_EXPRESSIONS = Set.of(
    "pandas.core.frame.DataFrame.drop",
    "pandas.core.frame.DataFrame.dropna",
    "pandas.core.frame.DataFrame.drop_duplicates",
    "pandas.core.frame.DataFrame.sort_values",
    "pandas.core.frame.DataFrame.sort_index",
    "pandas.core.frame.DataFrame.query",
    "pandas.core.frame.DataFrame.transpose",
    "pandas.core.frame.DataFrame.swapaxes",
    "pandas.core.frame.DataFrame.reindex",
    "pandas.core.frame.DataFrame.reindex_like",
    "pandas.core.frame.DataFrame.truncate");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, PandasModifyInPlaceCheck::checkInplaceParameter);
  }

  private static void checkInplaceParameter(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Optional.ofNullable(callExpression.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .filter(FULLY_QUALIFIED_EXPRESSIONS::contains)
      .map(fqn -> TreeUtils.argumentByKeyword("inplace", callExpression.arguments()))
      .map(RegularArgument::expression)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .map(Name::name)
      .filter("True"::equals)
      .ifPresent(fqn -> ctx.addIssue(callExpression, MESSAGE));
  }
}
