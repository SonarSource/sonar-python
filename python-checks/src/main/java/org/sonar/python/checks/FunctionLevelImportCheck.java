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

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.semantic.BuiltinSymbols;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S9458")
public class FunctionLevelImportCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Replace this function-level import with a module-level \"lazy import\".";

  // Functions serialized by value and run in another process: a module-level lazy import is not available there
  private static final TypeMatcher REMOTE_EXECUTION_DECORATOR = TypeMatchers.any(Stream.of(
      "pyspark.sql.functions.udf",
      "pyspark.sql.functions.pandas_udf",
      "ray.remote",
      "dask.delayed",
      "dask.delayed.delayed",
      "airflow.decorators.task.virtualenv",
      "airflow.decorators.task.external_python",
      "airflow.sdk.task.virtualenv",
      "airflow.sdk.task.external_python")
    .map(TypeMatchers::withFQN));

  // Classes whose methods run in another process (Ray actors)
  private static final TypeMatcher REMOTE_ACTOR_DECORATOR = TypeMatchers.withFQN("ray.remote");

  private Set<String> moduleLevelNames = Set.of();
  private Set<String> namesBoundToDifferentModules = Set.of();
  private boolean lazyImportsSupported = false;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::startFile);
    context.registerSyntaxNodeConsumer(Tree.Kind.IMPORT_NAME, ctx -> {
      ImportName importName = (ImportName) ctx.syntaxNode();
      checkImport(ctx, importName, importName.modules());
    });
    context.registerSyntaxNodeConsumer(Tree.Kind.IMPORT_FROM, ctx -> {
      ImportFrom importFrom = (ImportFrom) ctx.syntaxNode();
      if (!importFrom.isWildcardImport()) {
        checkImport(ctx, importFrom, importFrom.importedNames());
      }
    });
  }

  private void startFile(SubscriptionContext ctx) {
    lazyImportsSupported = PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(ctx.sourcePythonVersions(), PythonVersionUtils.Version.V_315);
    if (!lazyImportsSupported) {
      return;
    }
    FileInput fileInput = (FileInput) ctx.syntaxNode();
    moduleLevelNames = fileInput.globalVariables().stream().map(Symbol::name).collect(Collectors.toSet());
    namesBoundToDifferentModules = namesBoundToDifferentModules(fileInput);
  }

  /**
   * Names bound by function-level imports to different targets in different places (e.g. "import numpy as xp" and
   * "import cupy as xp"): they cannot all be moved to the module level.
   */
  private static Set<String> namesBoundToDifferentModules(FileInput fileInput) {
    Map<String, Set<String>> targetsByName = new HashMap<>();
    fileInput.accept(new BaseTreeVisitor() {
      @Override
      public void visitImportName(ImportName importName) {
        importName.modules().forEach(module -> {
          String target = module.alias() != null ? dottedName(module.dottedName()) : boundName(module).name();
          addTarget(importName, module, target);
        });
      }

      @Override
      public void visitImportFrom(ImportFrom importFrom) {
        DottedName module = importFrom.module();
        String prefix = importFrom.dottedPrefixForModule().stream().map(Token::value).collect(Collectors.joining())
          + (module != null ? dottedName(module) : "");
        importFrom.importedNames().forEach(name -> addTarget(importFrom, name, prefix + ":" + dottedName(name.dottedName())));
      }

      private void addTarget(Statement importStatement, AliasedName importedName, String target) {
        if (TreeUtils.firstAncestorOfKind(importStatement, Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF) instanceof FunctionDef) {
          targetsByName.computeIfAbsent(boundName(importedName).name(), name -> new HashSet<>()).add(target);
        }
      }
    });
    return targetsByName.entrySet().stream()
      .filter(entry -> entry.getValue().size() > 1)
      .map(Map.Entry::getKey)
      .collect(Collectors.toSet());
  }

  private static String dottedName(DottedName dottedName) {
    return dottedName.names().stream().map(Name::name).collect(Collectors.joining("."));
  }

  private void checkImport(SubscriptionContext ctx, Statement importStatement, List<AliasedName> importedNames) {
    if (!lazyImportsSupported
      || !(TreeUtils.firstAncestorOfKind(importStatement, Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF) instanceof FunctionDef function)
      || isInsideTryOrWith(importStatement, function)
      || isRunInAnotherProcess(function, ctx)
      || !importedNames.stream().map(FunctionLevelImportCheck::boundName).allMatch(this::canBeMovedToModuleLevel)) {
      return;
    }
    ctx.addIssue(importStatement, MESSAGE);
  }

  private static Name boundName(AliasedName importedName) {
    Name alias = importedName.alias();
    return alias != null ? alias : importedName.dottedName().names().get(0);
  }

  /**
   * The bound name must be read in the function (otherwise the import only runs for its side effects), bound only by
   * the import, not already bound at module level, not a built-in name (a module-level binding would shadow the
   * built-in in the whole module), and not bound to a different module elsewhere.
   */
  private boolean canBeMovedToModuleLevel(Name boundName) {
    SymbolV2 symbol = boundName.symbolV2();
    String name = boundName.name();
    if (symbol == null || moduleLevelNames.contains(name) || BuiltinSymbols.all().contains(name) || namesBoundToDifferentModules.contains(name)) {
      return false;
    }
    List<UsageV2> usages = symbol.usages();
    return usages.stream().filter(UsageV2::isBindingUsage).count() == 1
      && usages.stream().anyMatch(usage -> !usage.isBindingUsage());
  }

  /**
   * The exception handling of a "try" statement, or the context set up by a "with" statement, would no longer apply to
   * a lazy import, which runs when the name is first used.
   */
  private static boolean isInsideTryOrWith(Tree importStatement, FunctionDef function) {
    for (Tree ancestor = importStatement.parent(); ancestor != null && ancestor != function; ancestor = ancestor.parent()) {
      if (ancestor.is(Tree.Kind.TRY_STMT, Tree.Kind.WITH_STMT)) {
        return true;
      }
    }
    return false;
  }

  /**
   * The function, an enclosing function, or the class of a method (a Ray actor) is serialized and run in another process.
   */
  private static boolean isRunInAnotherProcess(FunctionDef function, SubscriptionContext ctx) {
    for (Tree tree = function; tree != null; tree = TreeUtils.firstAncestorOfKind(tree, Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF)) {
      List<Decorator> decorators = tree instanceof FunctionDef functionDef ? functionDef.decorators() : ((ClassDef) tree).decorators();
      TypeMatcher matcher = tree instanceof FunctionDef ? REMOTE_EXECUTION_DECORATOR : REMOTE_ACTOR_DECORATOR;
      if (decorators.stream().anyMatch(decorator -> isDecoratedWith(decorator, matcher, ctx))) {
        return true;
      }
    }
    return false;
  }

  private static boolean isDecoratedWith(Decorator decorator, TypeMatcher matcher, SubscriptionContext ctx) {
    Expression expression = decorator.expression();
    Expression decoratorFunction = expression instanceof CallExpression call ? call.callee() : expression;
    return matcher.isTrueFor(decoratorFunction, ctx);
  }
}
