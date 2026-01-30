/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
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
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8385")
public class FlaskSendFileMimeTypeCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Provide \"mimetype\" or \"download_name\" when calling \"send_file\" with a file-like object.";
  private static final TypeMatcher SEND_FILE_MATCHER = TypeMatchers.isType("flask.send_file");
  private static final TypeMatcher FILE_LIKE_MATCHER = TypeMatchers.isObjectInstanceOf("typing.IO");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, FlaskSendFileMimeTypeCheck::checkCallExpression);
  }

  private static void checkCallExpression(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();

    if (!SEND_FILE_MATCHER.isTrueFor(callExpression.callee(), ctx)) {
      return;
    }

    List<Argument> arguments = callExpression.arguments();
    if (arguments.isEmpty()) {
      return;
    }

    RegularArgument firstArgument = TreeUtils.nthArgumentOrKeyword(0, "path_or_file", arguments);
    if (firstArgument == null || !FILE_LIKE_MATCHER.isTrueFor(firstArgument.expression(), ctx)) {
      return;
    }

    var mimetype = TreeUtils.nthArgumentOrKeyword(1, "mimetype", arguments);
    var fileName = TreeUtils.nthArgumentOrKeyword(3, "download_name", arguments);
    var fileNameLegacy = TreeUtils.nthArgumentOrKeyword(3, "attachment_filename", arguments);

    if (mimetype == null && fileName == null && fileNameLegacy == null) {
      ctx.addIssue(callExpression.callee(), MESSAGE);
    }
  }
}
