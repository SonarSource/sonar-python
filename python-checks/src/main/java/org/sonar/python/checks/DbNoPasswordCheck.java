/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;

import static org.sonar.plugins.python.api.tree.Tree.Kind.ASSIGNMENT_STMT;
import static org.sonar.plugins.python.api.tree.Tree.Kind.CALL_EXPR;
import static org.sonar.plugins.python.api.tree.Tree.Kind.DICTIONARY_LITERAL;
import static org.sonar.plugins.python.api.tree.Tree.Kind.FILE_INPUT;
import static org.sonar.plugins.python.api.tree.Tree.Kind.KEY_VALUE_PAIR;
import static org.sonar.plugins.python.api.tree.Tree.Kind.REGULAR_ARGUMENT;
import static org.sonar.plugins.python.api.tree.Tree.Kind.STATEMENT_LIST;
import static org.sonar.plugins.python.api.tree.Tree.Kind.STRING_LITERAL;

@Rule(key = "S2115")
public class DbNoPasswordCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Add password protection to this database.";

  private static final List<String> CONNECT_METHODS = Arrays.asList(
    "mysql.connector.connect",
    "mysql.connector.connection.MySQLConnection",
    "pymysql.connections.connect",
    "psycopg2.connect",
    "pgdb.connect.connect",
    "pg.DB",
    "pg.connect"
  );

  private static final Pattern CONNECTION_URI_PATTERN =
    Pattern.compile("^(?:postgresql|mysql|oracle|mssql)(?:\\+.+?)?://.+?(:.*)?@.+");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(STRING_LITERAL, DbNoPasswordCheck::checkDbUri);
    context.registerSyntaxNodeConsumer(KEY_VALUE_PAIR, DbNoPasswordCheck::checkDjangoSettings);
    context.registerSyntaxNodeConsumer(CALL_EXPR, DbNoPasswordCheck::checkDbApi);
  }

  private static void checkDbUri(SubscriptionContext ctx) {
    String value = ((StringLiteral) ctx.syntaxNode()).trimmedQuotesValue();
    Matcher matcher = CONNECTION_URI_PATTERN.matcher(value);
    if (matcher.find()) {
      String password = matcher.group(1);

      if (password == null || password.equals(":")) {
        ctx.addIssue(ctx.syntaxNode(), MESSAGE);
      }
    }
  }

  private static void checkDbApi(SubscriptionContext ctx) {
    CallExpression callExpr = (CallExpression) ctx.syntaxNode();
    Symbol symbol = callExpr.calleeSymbol();
    if (symbol != null && CONNECT_METHODS.contains(symbol.fullyQualifiedName())) {
      RegularArgument passwordArgument = getPasswordArgument(symbol.fullyQualifiedName(), callExpr.arguments());
      if (passwordArgument != null && isString(passwordArgument.expression(), "")) {
        ctx.addIssue(passwordArgument, MESSAGE);
      }
    }
  }

  private static RegularArgument getPasswordArgument(String method, List<Argument> arguments) {
    boolean isPg = method.startsWith("pg.");
    String argumentKeyword = isPg ? "passwd" : "password";
    int passwordIndex = isPg ? 5 : 2;
    int positionalIndex = 0;

    for (Argument argument : arguments) {
      if (!argument.is(REGULAR_ARGUMENT)) {
        continue;
      }
      RegularArgument regularArgument = (RegularArgument) argument;
      Name keyword = regularArgument.keywordArgument();

      if (keyword != null && keyword.name().equals(argumentKeyword)) {
        return regularArgument;
      } else if (keyword == null) {
        if (positionalIndex == passwordIndex) {
          return regularArgument;
        }
        positionalIndex++;
      }
    }
    return null;
  }

  private static void checkDjangoSettings(SubscriptionContext ctx) {
    if (!ctx.pythonFile().fileName().equals("settings.py")) {
      return;
    }
    KeyValuePair keyValue = (KeyValuePair) ctx.syntaxNode();

    Kind[] parentChain = {
      DICTIONARY_LITERAL,
      KEY_VALUE_PAIR,
      DICTIONARY_LITERAL,
      ASSIGNMENT_STMT,
      STATEMENT_LIST,
      FILE_INPUT};
    int index = 0;
    Tree currentParent = keyValue.parent();
    while (currentParent != null && currentParent.is(parentChain[index])) {
      if (currentParent.is(ASSIGNMENT_STMT) && !isDatabasesAssignment((AssignmentStatement) currentParent)) {
        return;
      }
      index++;
      currentParent = currentParent.parent();
    }

    if (index == parentChain.length
      && keyValue.key().is(STRING_LITERAL)
      && isString(keyValue.key(), "PASSWORD")
      && isString(keyValue.value(), "")) {
      ctx.addIssue(keyValue, MESSAGE);
    }
  }

  private static boolean isString(Tree tree, String value) {
    if (tree.is(STRING_LITERAL)) {
      return ((StringLiteral) tree).trimmedQuotesValue().equals(value);
    }
    return false;
  }

  private static boolean isDatabasesAssignment(AssignmentStatement assignment) {
    List<ExpressionList> lhs = assignment.lhsExpressions();
    return lhs.size() == 1 && lhs.get(0).expressions().size() == 1
      && lhs.get(0).expressions().get(0).is(Kind.NAME)
      && ((Name) lhs.get(0).expressions().get(0)).name().equals("DATABASES");
  }
}
