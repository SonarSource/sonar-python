/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.semantic.v2.callgraph;

import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ModuleType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.UnknownType.UnresolvedImportType;
import org.sonar.python.tree.TreeUtils;

public class CallGraphCollector {
  private CallGraphCollector() {
  }

  public static CallGraph collectCallGraph(FileInput rootTree) {
    var visitor = new Visitor();
    rootTree.accept(visitor);
    return visitor.build();
  }
    
  private static class Visitor extends BaseTreeVisitor {
    private final CallGraph.Builder callGraphBuilder = new CallGraph.Builder();

    @Override
    public void visitCallExpression(CallExpression callExpr) {
      super.visitCallExpression(callExpr);
      getCalledFunctionFqn(callExpr).ifPresent(calledFunctionFqn -> 
        getEnclosedFunctionFqn(callExpr).ifPresent(enclosedFunctionFqn -> 
          callGraphBuilder.addUsage(enclosedFunctionFqn, calledFunctionFqn)
        )
      );
    }

    private static Optional<String> getCalledFunctionFqn(CallExpression callExpr) {
      var calleeType = callExpr.callee().typeV2();
      return CallGraphCollector.getFqn(calleeType);
    }

    private static Optional<String> getEnclosedFunctionFqn(CallExpression callExpr) {
      Tree enclosingFuncDefTree = TreeUtils.firstAncestorOfKind(callExpr, Tree.Kind.FUNCDEF, Tree.Kind.LAMBDA);
      if(enclosingFuncDefTree instanceof FunctionDef enclosingFunctionDef) {
        return CallGraphCollector.getFqn(enclosingFunctionDef.name().typeV2());
      }
      // lambdas are not supported; thus returning empty
      return Optional.empty();
    }

    public CallGraph build() {
      return callGraphBuilder.build();
    }
  }

  private static Optional<String> getFqn(PythonType type) {
    if(type instanceof FunctionType functionType) {
      return Optional.of(functionType.fullyQualifiedName());
    } else if (type instanceof ModuleType moduleType) {
      return Optional.of(moduleType.fullyQualifiedName());
    } else if(type instanceof UnresolvedImportType unresolvedImportType) {
      return Optional.of(unresolvedImportType.importPath());
    } else if(type instanceof UnionType unionType) {
      Set<String> unionFqnSet = unionType.candidates().stream()
        .flatMap(candidate -> getFqn(candidate).stream())
        .collect(Collectors.toSet());
      
      if (unionFqnSet.size() == 1) {
        return unionFqnSet.stream().findFirst();
      } else {
        // Multiple candidates, cannot determine a single FQN
        return Optional.empty(); 
      }
    } else {
      return Optional.empty();
    }
  }
}
