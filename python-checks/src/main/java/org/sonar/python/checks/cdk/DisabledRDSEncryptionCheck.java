/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.checks.cdk;

import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S6303")
public class DisabledRDSEncryptionCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    new DisabledRDSEncryptionDatabaseClusterCheck().initialize(context);
    new DisabledRDSEncryptionDatabaseInstanceCheck().initialize(context);
    new DisabledRDSEncryptionCfnDBClusterCheck().initialize(context);
    new DisabledRDSEncryptionCfnDBInstanceCheck().initialize(context);
  }

  // DatabaseCluster and DatabaseInstance
  abstract static class DisabledRDSEncryptionDatabaseCheck extends AbstractCdkResourceCheck {
    private static final String ARG_ENCRYPTED = "storage_encrypted";
    private static final String ARG_ENCRYPTION_KEY = "storage_encryption_key";
    private static final String OMITTING_MESSAGE = "Omitting \""+ARG_ENCRYPTED+"\" and \""+ARG_ENCRYPTION_KEY+"\" disables RDS encryption. Make sure it is safe here.";
    private static final String UNENCRYPTED_MESSAGE = "Make sure that using unencrypted databases is safe here.";

    @Override
    protected void visitResourceConstructor(SubscriptionContext ctx, CallExpression resourceConstructor) {
      Map<String, ArgumentTrace> map = getArguments(ctx, resourceConstructor,
        List.of(ARG_ENCRYPTED, ARG_ENCRYPTION_KEY));
      ArgumentTrace argEncrypted = map.get(ARG_ENCRYPTED);
      ArgumentTrace argEncryptionKey = map.get(ARG_ENCRYPTION_KEY);

      if (argEncrypted == null && argEncryptionKey == null) {
        ctx.addIssue(resourceConstructor.callee(), OMITTING_MESSAGE);
      } else if (argEncrypted == null) {
        argEncryptionKey.addIssueIf(AbstractCdkResourceCheck::isNone, UNENCRYPTED_MESSAGE);
      } else if (argEncryptionKey == null) {
        argEncrypted.addIssueIf(AbstractCdkResourceCheck::isFalse, UNENCRYPTED_MESSAGE);
      } else {
        if (argEncryptionKey.hasExpression(AbstractCdkResourceCheck::isNone)
          && argEncrypted.hasExpression(AbstractCdkResourceCheck::isFalse)) {
          argEncrypted.addIssue(UNENCRYPTED_MESSAGE);
        }
      }
    }
  }

  static class DisabledRDSEncryptionDatabaseClusterCheck extends DisabledRDSEncryptionDatabaseCheck {
    @Override
    protected String resourceFqn() {
      return "aws_cdk.aws_rds.DatabaseCluster";
    }
  }

  static class DisabledRDSEncryptionDatabaseInstanceCheck extends DisabledRDSEncryptionDatabaseCheck {
    @Override
    protected String resourceFqn() {
      return "aws_cdk.aws_rds.DatabaseInstance";
    }
  }

  // CfnDBCluster and CfnDBInstance
  abstract static class DisabledRDSEncryptionCfnDBCheck extends AbstractCdkResourceCheck {
    private static final String ARG_ENCRYPTED = "storage_encrypted";
    private static final String OMITTING_MESSAGE = "Omitting \""+ARG_ENCRYPTED+"\" disables RDS encryption. Make sure it is safe here.";
    private static final String UNENCRYPTED_MESSAGE = "Make sure that using unencrypted databases is safe here.";

    protected abstract boolean additionalCheck(SubscriptionContext ctx, CallExpression resourceConstructor);

    @Override
    protected void visitResourceConstructor(SubscriptionContext ctx, CallExpression resourceConstructor) {
      if (additionalCheck(ctx, resourceConstructor)) {
        getArgument(ctx, resourceConstructor, ARG_ENCRYPTED).ifPresentOrElse(
          argumentTrace -> argumentTrace.addIssueIf(AbstractCdkResourceCheck::isFalse, UNENCRYPTED_MESSAGE),
          () -> ctx.addIssue(resourceConstructor.callee(), OMITTING_MESSAGE)
        );
      }
    }
  }

  static class DisabledRDSEncryptionCfnDBClusterCheck extends DisabledRDSEncryptionCfnDBCheck {
    @Override
    protected String resourceFqn() {
      return "aws_cdk.aws_rds.CfnDBCluster";
    }

    @Override
    protected boolean additionalCheck(SubscriptionContext ctx, CallExpression resourceConstructor) {
      return true;
    }
  }

  static class DisabledRDSEncryptionCfnDBInstanceCheck extends DisabledRDSEncryptionCfnDBCheck {
    @Override
    protected String resourceFqn() {
      return "aws_cdk.aws_rds.CfnDBInstance";
    }

    @Override
    protected boolean additionalCheck(SubscriptionContext ctx, CallExpression resourceConstructor) {
      return getArgument(ctx, resourceConstructor, "engine")
        .filter(argumentTrace ->
          argumentTrace.hasExpression(expression ->
            Optional.of(expression)
              .filter(expr -> expr.is(Tree.Kind.STRING_LITERAL)).map(StringLiteral.class::cast)
              .filter(stringLiteral -> stringLiteral.trimmedQuotesValue().toLowerCase(Locale.ROOT).startsWith("aurora"))
              .isPresent()
          )
        ).isEmpty();
    }
  }
}
