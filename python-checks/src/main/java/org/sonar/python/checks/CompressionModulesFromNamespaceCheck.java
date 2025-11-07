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
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Tree.Kind;

import javax.annotation.Nullable;

@Rule(key = "S7941")
public class CompressionModulesFromNamespaceCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Compression modules should be imported from the compression namespace.";
  private static final Set<String> COMPRESSION_MODULES = Set.of("lzma", "bz2", "gzip", "zlib");

  private boolean isPython314OrGreater = false;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FILE_INPUT, this::initializeState);
    context.registerSyntaxNodeConsumer(Kind.IMPORT_FROM, this::checkImportFrom);
    context.registerSyntaxNodeConsumer(Kind.IMPORT_NAME, this::checkImportName);
  }

  private void initializeState(SubscriptionContext ctx) {
    isPython314OrGreater = PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(ctx.sourcePythonVersions(), PythonVersionUtils.Version.V_314);
  }

  private void checkImportFrom(SubscriptionContext ctx) {
    if (!isPython314OrGreater) {
      return;
    }
    ImportFrom importFrom = (ImportFrom) ctx.syntaxNode();
    DottedName dottedModuleName = importFrom.module();
    raiseIfSingleNameCompressionModule(ctx, dottedModuleName);
  }

  private void checkImportName(SubscriptionContext ctx) {
    if (!isPython314OrGreater) {
      return;
    }
    ImportName importName = (ImportName) ctx.syntaxNode();
    for (AliasedName aliasedModuleName : importName.modules()) {
      DottedName dottedModuleName = aliasedModuleName.dottedName();
      raiseIfSingleNameCompressionModule(ctx, dottedModuleName);
    }
  }

  private static void raiseIfSingleNameCompressionModule(SubscriptionContext ctx, @Nullable DottedName moduleName) {
    if (moduleName != null
        && moduleName.names() != null 
        && moduleName.names().size() == 1 
        && COMPRESSION_MODULES.contains(moduleName.names().get(0).name())
    ) {
      ctx.addIssue(moduleName, MESSAGE);
    }
  }

}
