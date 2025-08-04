/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.AwsLambdaChecksUtils;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S6243")
public class AwsLambdaClientInstantiationCheck extends PythonSubscriptionCheck {

  private static final String CLIENT_ISSUE_MESSAGE = "Initialize this AWS client outside the Lambda handler function.";
  private static final String DATABASE_ISSUE_MESSAGE = "Initialize this database connection outside the Lambda handler function.";
  private static final String ORM_ISSUE_MESSAGE = "Initialize this ORM connection outside the Lambda handler function.";

  private static final Set<String> CLIENT_FQNS = Set.of(
    "boto3.client",
    "boto3.resource",
    "boto3.session.Session"
  );

  private static final Set<String> DATABASE_FQNS = Set.of(
    "pymysql.connect",
    "mysql.connector.connect",
    "psycopg2.connect",
    "pymongo.MongoClient",
    "sqlite3.dbapi2.connect",
    "redis.Redis",
    "redis.StrictRedis"
  );

  private static final Set<String> ORM_FQNS = Set.of(
    "sqlalchemy.orm.sessionmaker",
    "peewee.PostgresqlDatabase",
    "peewee.MySQLDatabase",
    "peewee.SqliteDatabase",
    "mongoengine.connect"
  );

  private final TypeCheckMap<String> isClientOrResourceTypeCheckMap = new TypeCheckMap<>();

  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::setupTypeChecker);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCall);
  }

  private void setupTypeChecker(SubscriptionContext ctx) {
    CLIENT_FQNS.forEach(fqn -> isClientOrResourceTypeCheckMap.put(ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(fqn), CLIENT_ISSUE_MESSAGE));
    DATABASE_FQNS.forEach(fqn -> isClientOrResourceTypeCheckMap.put(ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(fqn), DATABASE_ISSUE_MESSAGE));
    ORM_FQNS.forEach(fqn -> isClientOrResourceTypeCheckMap.put(ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(fqn), ORM_ISSUE_MESSAGE));
  }

  private void checkCall(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();

    if (!isInAWSLambdaFunction(callExpression, ctx)) {
      return;
    }

    String message = isClientOrResourceTypeCheckMap.getForType(callExpression.callee().typeV2());
    if (message != null) {
      ctx.addIssue(callExpression, message);
    }
  }

  private static boolean isInAWSLambdaFunction(CallExpression callExpression, SubscriptionContext ctx) {
    Tree parentFunctionDef = TreeUtils.firstAncestorOfKind(callExpression.parent(), Tree.Kind.FUNCDEF);
    if (parentFunctionDef == null) {
      return false;
    }
    return AwsLambdaChecksUtils.isLambdaHandler(ctx, (FunctionDef) parentFunctionDef);
  }
}


