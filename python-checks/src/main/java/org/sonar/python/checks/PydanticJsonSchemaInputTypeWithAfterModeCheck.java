/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks;

import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

/**
 * Detects usage of {@code json_schema_input_type} with {@code mode='after'} (or no mode, which defaults to 'after')
 * in Pydantic's {@code @field_validator} decorator. This combination always causes a {@code PydanticUserError}
 * at class definition time.
 */
@Rule(key = "S8974")
public class PydanticJsonSchemaInputTypeWithAfterModeCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove \"json_schema_input_type\" or change the validator mode to \"before\", \"plain\", or \"wrap\".";
  private static final String SECONDARY_MESSAGE = "mode is set here.";

  private static final String MODE_ARGUMENT = "mode";
  private static final String JSON_SCHEMA_INPUT_TYPE_ARGUMENT = "json_schema_input_type";
  private static final String AFTER_MODE_VALUE = "after";

  /**
   * Matches {@code pydantic.functional_validators.field_validator} (direct import):
   *   {@code from pydantic.functional_validators import field_validator}
   * <p>
   * Also covers aliased imports, e.g.:
   *   {@code from pydantic.functional_validators import field_validator as fv}
   */
  private static final TypeMatcher FIELD_VALIDATOR_DIRECT_MATCHER = TypeMatchers.isType("pydantic.functional_validators.field_validator");

  /**
   * Matches {@code field_validator} re-exported through the {@code pydantic} package root:
   *   {@code from pydantic import field_validator}
   * <p>
   * Uses FQN matching because the pydantic v1 stub does not declare this re-export,
   * so the type resolves to {@code UnresolvedImportType[pydantic.field_validator]}.
   * Also covers aliased imports, e.g.:
   *   {@code from pydantic import field_validator as fv}
   */
  private static final TypeMatcher FIELD_VALIDATOR_REEXPORT_MATCHER = TypeMatchers.withFQN("pydantic.field_validator");

  private static final TypeMatcher FIELD_VALIDATOR_MATCHER = TypeMatchers.any(FIELD_VALIDATOR_DIRECT_MATCHER, FIELD_VALIDATOR_REEXPORT_MATCHER);

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.DECORATOR, PydanticJsonSchemaInputTypeWithAfterModeCheck::checkDecorator);
  }

  private static void checkDecorator(SubscriptionContext ctx) {
    Decorator decorator = (Decorator) ctx.syntaxNode();
    Expression decoratorFunctionExpr = getDecoratorFunctionExpression(decorator);
    if (!FIELD_VALIDATOR_MATCHER.isTrueFor(decoratorFunctionExpr, ctx)) {
      return;
    }
    ArgList argList = decorator.arguments();
    if (argList == null || argList.arguments().isEmpty()) {
      return;
    }

    RegularArgument jsonSchemaArg = TreeUtils.argumentByKeyword(JSON_SCHEMA_INPUT_TYPE_ARGUMENT, argList.arguments());
    if (jsonSchemaArg == null) {
      return;
    }

    var keywordArgument = jsonSchemaArg.keywordArgument();
    if (keywordArgument == null) {
      return;
    }

    RegularArgument modeArg = TreeUtils.argumentByKeyword(MODE_ARGUMENT, argList.arguments());
    if (!isAfterModeOrDefault(modeArg)) {
      return;
    }

    PythonCheck.PreciseIssue issue = ctx.addIssue(keywordArgument, MESSAGE);
    if (modeArg != null) {
      issue.secondary(modeArg.expression(), SECONDARY_MESSAGE);
    }
  }

  private static Expression getDecoratorFunctionExpression(Decorator decorator) {
    Expression expr = decorator.expression();
    if (expr instanceof CallExpression callExpr) {
      return callExpr.callee();
    }
    return expr;
  }

  /**
   * Returns {@code true} if {@code mode} is absent (defaults to {@code 'after'}) or explicitly set to {@code 'after'}.
   * Returns {@code false} for {@code 'before'}, {@code 'plain'}, {@code 'wrap'}, or any non-literal value.
   */
  private static boolean isAfterModeOrDefault(@CheckForNull RegularArgument modeArg) {
    if (modeArg == null) {
      // No mode argument — defaults to 'after'
      return true;
    }
    return isAfterLiteralValue(modeArg.expression());
  }

  private static boolean isAfterLiteralValue(Expression expression) {
    // Only flag when mode is explicitly 'after'; unknown (non-literal) mode values → no issue
    if (expression instanceof StringLiteral stringLiteral) {
      return AFTER_MODE_VALUE.equals(stringLiteral.trimmedQuotesValue());
    }
    return false;
  }
}
