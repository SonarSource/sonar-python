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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeParams;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5953")
public class UndefinedSymbolsCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      FileInput fileInput = (FileInput) ctx.syntaxNode();
      if (importsManipulatedAllProperty(fileInput)) {
        return;
      }
      UnresolvedSymbolsVisitor unresolvedSymbolsVisitor = new UnresolvedSymbolsVisitor();
      fileInput.accept(unresolvedSymbolsVisitor);
      if (!unresolvedSymbolsVisitor.callGlobalsOrLocals && !unresolvedSymbolsVisitor.hasWildcardImport) {
        addNameIssues(unresolvedSymbolsVisitor.nameIssues, ctx);
      }
    });
  }

  private static boolean importsManipulatedAllProperty(FileInput fileInput) {
    return fileInput.globalVariables().stream().anyMatch(s -> "__all__".equals(s.name()) && s.fullyQualifiedName() != null);
  }

  private static void addNameIssues(Map<String, List<Name>> nameIssues, SubscriptionContext subscriptionContext) {
    nameIssues.forEach((name, list) -> {
      Name first = list.get(0);
      PreciseIssue issue = subscriptionContext.addIssue(first, first.name() + " is not defined. Change its name or define it before using it");
      list.stream().skip(1).forEach(n -> issue.secondary(n, null));
    });
  }

  private static class UnresolvedSymbolsVisitor extends BaseTreeVisitor {

    private boolean hasWildcardImport = false;
    private boolean callGlobalsOrLocals = false;
    private final Map<String, List<Name>> nameIssues = new HashMap<>();

    @Override
    public void visitName(Name name) {
      if (name.isVariable() && name.symbol() == null && !name.name().startsWith("_") && !isTypeVar(name)) {
        nameIssues.computeIfAbsent(name.name(), k -> new ArrayList<>()).add(name);
      }
    }

    private static boolean isTypeVar(Name name) {
      return TreeUtils.firstAncestor(name, tree -> classWithTypeVar(tree, name)) != null;
    }

    private static boolean classWithTypeVar(Tree tree, Name name) {
      if (tree instanceof ClassDef classDef) {
        return hasTypeVar(classDef.typeParams(), name);
      }
      return false;
    }

    private static boolean hasTypeVar(@Nullable TypeParams typeParams, Name name) {
      if (typeParams != null && !typeParams.typeParamsList().isEmpty()) {
        return typeParams.typeParamsList().stream().anyMatch(typeParam -> typeParam.name().name().equals(name.name()));
      }
      return false;
    }

    @Override
    public void visitImportFrom(ImportFrom importFrom) {
      hasWildcardImport |= importFrom.wildcard() != null;
      super.visitImportFrom(importFrom);
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      if (callExpression.callee().is(Tree.Kind.NAME)) {
        String name = ((Name) callExpression.callee()).name();
        callGlobalsOrLocals |= "globals".equals(name) || "locals".equals(name);
      }
      super.visitCallExpression(callExpression);
    }
  }
}
