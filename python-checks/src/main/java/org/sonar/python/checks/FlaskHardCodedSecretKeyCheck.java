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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

@Rule(key = "S6779")
public class FlaskHardCodedSecretKeyCheck extends FlaskHardCodedSecret {

  private static final String SECRET_KEY_KEYWORD = "SECRET_KEY";
  private static final String SECRET_KEY_TYPE = "\"Flask\"";
  private static final TypeMatcher FLASK_APP_MATCHER = TypeMatchers.isObjectInstanceOf("flask.app.Flask");

  @Override
  protected String getSecretKeyKeyword() {
    return SECRET_KEY_KEYWORD;
  }

  @Override
  protected String getSecretKeyType() {
    return SECRET_KEY_TYPE;
  }

  @Override
  protected boolean isSensitiveProperty(Expression expression, SubscriptionContext ctx) {
    if (expression.is(Tree.Kind.QUALIFIED_EXPR)) {
      QualifiedExpression qualifiedExpression = (QualifiedExpression) expression;
      return "secret_key".equals(qualifiedExpression.name().name())
        && FLASK_APP_MATCHER.isTrueFor(qualifiedExpression.qualifier(), ctx);
    }
    return isFlaskConfigSecret(expression);
  }

}
