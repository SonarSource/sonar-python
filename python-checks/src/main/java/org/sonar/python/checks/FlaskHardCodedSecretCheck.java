package org.sonar.python.checks;

import java.util.List;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;


@Rule(key = "S6779")
public class FlaskHardCodedSecretCheck extends PythonSubscriptionCheck {
  private static final Logger LOG = LoggerFactory.getLogger(FlaskHardCodedSecretCheck.class);


  private static final String MESSAGE = "Don't disclose \"Flask\" secret keys.";
  private static final String SECRET_KEY_KEYWORD = "SECRET_KEY";
  private static final Set<String> FLASK_APP_CONFIG_UPDATE_CALLEE_QUALIFIER_FQNS = Set.of(
    "flask.globals.current_app.config",
    "flask.app.Flask.config"
  );

  @Override
  public void initialize(Context context) {


    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, FlaskHardCodedSecretCheck::verifyCallExpression);
//    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, FlaskHardCodedSecretCheck::verifyAssignmentStatement);
//    context.registerSyntaxNodeConsumer(Tree.Kind.QUALIFIED_EXPR, FlaskHardCodedSecretCheck::verifyQualifiedExpression);
  }


  private static void verifyCallExpression(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    LOG.info("Consuming Call Expression:");
    Optional.of(callExpression)
      .map(CallExpression::callee)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .filter(qualiExpr -> "update".equals(qualiExpr.name().name()))
      .map(QualifiedExpression::qualifier)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .map(QualifiedExpression::name)
      .map(Name::symbol)
      .map(Symbol::fullyQualifiedName)
      .filter(FLASK_APP_CONFIG_UPDATE_CALLEE_QUALIFIER_FQNS::contains)
      .ifPresent(fqn -> {
          verifyUpdateCallArgument(ctx, callExpression);
          LOG.info(String.format("Qualifer fqn=%s, at line %d", fqn, TreeUtils.locationInFile(callExpression,
            ctx.pythonFile().fileName()).startLine()));
        }
      );

    LOG.info("");
  }

  private static void verifyUpdateCallArgument(SubscriptionContext ctx, CallExpression callExpression) {
    Optional.of(callExpression.arguments())
      .filter(arguments -> arguments.size() == 1)
      .map(arguments -> arguments.get(0))
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(RegularArgument.class))
      .map(RegularArgument::expression)
      .map(FlaskHardCodedSecretCheck::getAssignedValue)
      .filter(FlaskHardCodedSecretCheck::isIllegalDictArgument)
      .ifPresent(expr -> ctx.addIssue(callExpression, MESSAGE));

  }

  private static Expression getAssignedValue(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      return Expressions.singleAssignedValue((Name) expression);
    }
    return expression;
  }

  private static boolean isIllegalDictArgument(Expression expression) {
    if (expression.is(Tree.Kind.CALL_EXPR)) {
      return isCallToDictConstructor((CallExpression) expression) && hasIllegalKeywordArgument((CallExpression) expression);
    } else if (expression.is(Tree.Kind.DICTIONARY_LITERAL)) {
      return hasIllegalKeyValuePair((DictionaryLiteral) expression);
    }
    return false;
  }


  private static boolean isCallToDictConstructor(CallExpression callExpression) {
    return Optional.of(callExpression)
      .map(CallExpression::callee)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .map(Name::symbol)
      .map(Symbol::fullyQualifiedName)
      .filter("dict"::equals)
      .isPresent();
  }


  private static boolean hasIllegalKeyValuePair(DictionaryLiteral dictionaryLiteral) {
    return dictionaryLiteral.elements().stream()
      .filter(KeyValuePair.class::isInstance)
      .map(KeyValuePair.class::cast)
      .map(KeyValuePair::key)
      .filter(StringLiteral.class::isInstance)
      .map(StringLiteral.class::cast)
      .map(StringLiteral::trimmedQuotesValue)
      .anyMatch(SECRET_KEY_KEYWORD::equals);
  }

  private static boolean hasIllegalKeywordArgument(CallExpression callExpression) {
    return Optional.ofNullable(TreeUtils.argumentByKeyword(SECRET_KEY_KEYWORD, callExpression.arguments()))
      .map(RegularArgument::expression)
      .filter(FlaskHardCodedSecretCheck::isStringLiteral)
      .isPresent();
  }


  private static void verifyAssignmentStatement(SubscriptionContext ctx) {
    AssignmentStatement assignmentStatementTree = (AssignmentStatement) ctx.syntaxNode();
    boolean isAssigningDebugProperties = assignmentStatementTree.lhsExpressions()
      .stream()
      .map(ExpressionList::expressions)
      .flatMap(List::stream)
      .anyMatch(FlaskHardCodedSecretCheck::isSensitiveProperty);
  }

  private static boolean isSensitiveProperty(Expression expression) {
    return true;
  }

  private static void verifyQualifiedExpression(SubscriptionContext ctx) {
    QualifiedExpression callExpression = (QualifiedExpression) ctx.syntaxNode();
    LOG.info("Consuming Qualified Expression:");
    Optional.of(callExpression)
      .map(QualifiedExpression::symbol)
      .map(Symbol::fullyQualifiedName)
      .map("QUALIFIED EXPRESSION: "::concat)
      .ifPresentOrElse(LOG::info, () -> LOG.info("No fully Qualified name"));
    LOG.info("");
  }

  private static void verifySubscriptionExpression(SubscriptionContext ctx) {
    SubscriptionExpression subscriptionExpression = (SubscriptionExpression) ctx.syntaxNode();
    LOG.info("Consuming Subscription Expression:");
    Optional.of(subscriptionExpression)
      .map(SubscriptionExpression::object)
      .filter(HasSymbol.class::isInstance)
      .map(HasSymbol.class::cast)
      .map(HasSymbol::symbol)
      .map(Symbol::fullyQualifiedName)
//      .filter(CURR_APP_CONFIG_SUBSCRIPTION_FQN::equals)
      .map("SUBSCRIPTION EXPRESSION: "::concat)
      .ifPresentOrElse(LOG::info, () -> LOG.info("No fully Qualified name"));
//      .ifPresent(LOG::info);
    LOG.info("");
  }

  private static boolean isStringLiteral(@Nullable Expression expr) {
    if (expr == null) {
      return false;
    } else if (expr.is(Tree.Kind.STRING_LITERAL)) {
      return true;
    } else if (expr.is(Tree.Kind.NAME)) {
      return isStringLiteral(Expressions.singleAssignedValue((Name) expr));
    }
    return false;
  }
}
