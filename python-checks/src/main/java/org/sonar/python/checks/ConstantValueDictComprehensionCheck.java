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

import java.util.List;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.DictCompExpression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S7519")
public class ConstantValueDictComprehensionCheck extends PythonSubscriptionCheck {
  public static final String MESSAGE = "Replace with dict fromkeys method call";


  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.DICT_COMPREHENSION, ConstantValueDictComprehensionCheck::checkDictComprehension);
  }

  private static void checkDictComprehension(SubscriptionContext ctx) {
    var dictComprehension = (DictCompExpression) ctx.syntaxNode();
    if (isConstantValueDictComprehension(dictComprehension)) {
      ctx.addIssue(dictComprehension, MESSAGE);
    }
  }

  private static boolean isConstantValueDictComprehension(DictCompExpression dictComprehension) {
    if (!(dictComprehension.resultElement() instanceof KeyValuePair pair)
        || !(pair.key() instanceof Name)
        || dictComprehension.comprehensionFor().nestedClause() != null) {
      return false;
    }

    if (pair.value() instanceof Name valueName) {
      var valueSymbol = valueName.symbolV2();
      return valueSymbol == null || !valueSymbol
        .usages()
        .stream()
        .map(UsageV2::tree)
        .allMatch(ut -> TreeUtils.firstAncestor(ut, dictComprehension::equals) != null);
    } else if (pair.value() instanceof StringLiteral stringLiteral) {
      return stringLiteral.stringElements().stream().map(StringElement::formattedExpressions).allMatch(List::isEmpty);
    } else {
      return pair.value().is(Tree.Kind.NONE, Tree.Kind.NUMERIC_LITERAL, Tree.Kind.BOOLEAN_LITERAL_PATTERN);
    }
  }
}
