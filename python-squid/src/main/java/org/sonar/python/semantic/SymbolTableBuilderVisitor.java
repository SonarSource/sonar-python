/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.python.semantic;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.PythonVisitor;
import org.sonar.python.PythonVisitorContext;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.sslr.ast.AstSelect;

public class SymbolTableBuilderVisitor extends PythonVisitor {

  private Map<AstNode, Scope> scopesByRootTree;
  private Set<AstNode> allReadUsages;
  private Map<String, Module> importedModules = new HashMap<>();

  public SymbolTable symbolTable() {
    return new SymbolTablImpl(scopesByRootTree);
  }

  @Override
  public void scanFile(PythonVisitorContext context) {
    super.scanFile(context);
    new FirstPhaseVisitor().scanFile(context);
    new SecondPhaseVisitor().scanFile(context);
  }

  @Override
  public void visitFile(AstNode node) {
    scopesByRootTree = new HashMap<>();
    allReadUsages = new HashSet<>();
  }

  private static class ScopeVisitor extends PythonVisitor {

    private Deque<AstNode> scopeRootTrees = new LinkedList<>();

    @Override
    public void visitFile(AstNode node) {
      enterScope(node);
    }

    public void enterScope(AstNode node) {
      scopeRootTrees.push(node);
    }

    @Override
    public void visitNode(AstNode node) {
      if (node.is(PythonGrammar.FUNCDEF, PythonGrammar.CLASSDEF, PythonGrammar.LAMBDEF, PythonGrammar.LAMBDEF_NOCOND)) {
        enterScope(node);
      }
    }

    @Override
    public void leaveNode(AstNode node) {
      if (node.is(PythonGrammar.FUNCDEF, PythonGrammar.CLASSDEF, PythonGrammar.LAMBDEF, PythonGrammar.LAMBDEF_NOCOND)) {
        scopeRootTrees.pop();
      }
    }

    public AstNode currentScopeRootTree() {
      return scopeRootTrees.peek();
    }

  }

  private class FirstPhaseVisitor extends ScopeVisitor {

    @Override
    public Set<AstNodeType> subscribedKinds() {
      Set<AstNodeType> set = new HashSet<>();
      set.add(PythonGrammar.FUNCDEF);
      set.add(PythonGrammar.LAMBDEF);
      set.add(PythonGrammar.LAMBDEF_NOCOND);
      set.add(PythonGrammar.FOR_STMT);
      set.add(PythonGrammar.COMP_FOR);
      set.add(PythonGrammar.CLASSDEF);
      set.add(PythonGrammar.EXPRESSION_STMT);
      set.add(PythonGrammar.GLOBAL_STMT);
      set.add(PythonGrammar.NONLOCAL_STMT);
      set.add(PythonGrammar.IMPORT_STMT);
      set.add(PythonGrammar.CALL_EXPR);
      return Collections.unmodifiableSet(set);
    }

    @Override
    public void visitFile(AstNode node) {
      super.visitFile(node);
      createScope(node, null);
    }

    @Override
    public void visitNode(AstNode node) {
      Scope currentScope = currentScope();

      super.visitNode(node);

      if (node.is(PythonGrammar.FUNCDEF)) {
        createScope(node, currentScope);
        createFunctionParameters(node);

      } else if (node.is(PythonGrammar.LAMBDEF, PythonGrammar.LAMBDEF_NOCOND)) {
        createScope(node, currentScope);
        createLambdaParameters(node);

      } else if (node.is(PythonGrammar.FOR_STMT, PythonGrammar.COMP_FOR)) {
        createLoopVariables(node);

      } else if (node.is(PythonGrammar.CLASSDEF)) {
        createScope(node, currentScope);

      } else if (node.is(PythonGrammar.EXPRESSION_STMT)) {
        visitAssignment(node);

      } else if (node.is(PythonGrammar.GLOBAL_STMT)) {
        node.getChildren(PythonGrammar.NAME).forEach(name -> currentScope().addGlobalName(name.getTokenValue()));

      } else if (node.is(PythonGrammar.NONLOCAL_STMT)) {
        node.getChildren(PythonGrammar.NAME).forEach(name -> currentScope().addNonlocalName(name.getTokenValue()));

      } else if (node.is(PythonGrammar.IMPORT_STMT)) {
        visitImportStatement(node);

      } else if(node.is(PythonGrammar.CALL_EXPR)) {
        /* Given `myModuleName.f()` node, it adds `myModuleName.f` to the symbol table
           and resolves its name */
        AstNode attributeRef = node.getFirstChild(PythonGrammar.ATTRIBUTE_REF);
        if (attributeRef != null) {
          String namespace = attributeRef.getFirstChild(PythonGrammar.ATOM).getTokenValue();
          Module module = importedModules.get(namespace);
          if (module != null) {
            String functionName = attributeRef.getFirstChild(PythonGrammar.NAME).getTokenValue();
            currentScope().rootScope().addWriteUsage(attributeRef, namespace + "." + functionName, module.name);
          }
        }
      }
    }

    /**
     * Adds imported symbols to the symbol table.
     * Keeps track of imported modules and eventually of their aliases.
     *
     * ex: `import myModule` => add myModule to the symbol table
     * ex: `import myModule as foo` => add foo to the symbol table
     * ex: `from myModule import f => add f to the symbol table` with "myModule" as moduleName
     */
    private void visitImportStatement(AstNode importNode) {
      /* example: import myModule as foo */
      AstNode node = importNode.getFirstChild();
      if (node.is(PythonGrammar.IMPORT_NAME)) {
          node.getDescendants(PythonGrammar.DOTTED_AS_NAME).forEach(dottedAsName ->
            addImportedSymbols(
              dottedAsName.getFirstChild(PythonGrammar.DOTTED_NAME),
              dottedAsName.getFirstChild(PythonGrammar.NAME)));

        /* example: from myModule import f */
      } else if(node.is(PythonGrammar.IMPORT_FROM)) {
        String moduleName = node.getFirstChild(PythonGrammar.DOTTED_NAME).getTokenValue();
        node.getDescendants(PythonGrammar.IMPORT_AS_NAME).forEach(
          importAsName -> currentScope().addWriteUsage(importAsName.getFirstChild(PythonGrammar.NAME), moduleName));
      }
    }

    private void addImportedSymbols(AstNode moduleNameNode, @Nullable AstNode aliasNode) {
      Module module = new Module();
      module.name = moduleNameNode.getTokenValue();
      if (aliasNode != null) {
        currentScope().addWriteUsage(aliasNode);
        module.alias = aliasNode.getTokenValue();
        importedModules.put(module.alias, module);
      } else {
        currentScope().addWriteUsage(moduleNameNode);
        importedModules.put(module.name, module);
      }
    }

    private void visitAssignment(AstNode node) {
      for (AstNode assignOperator : node.getChildren(PythonPunctuator.ASSIGN, PythonGrammar.AUGASSIGN, PythonGrammar.ANNASSIGN)) {
        AstNode target = assignOperator.getPreviousSibling();
        if (assignOperator.is(PythonGrammar.ANNASSIGN)) {
          assignOperator = assignOperator.getFirstChild(PythonPunctuator.ASSIGN);
        }
        if (assignOperator != null) {
          if (currentScopeRootTree().is(PythonGrammar.CLASSDEF)) {
            FirstPhaseVisitor firstPhaseVisitor = new FirstPhaseVisitor();
            firstPhaseVisitor.enterScope(currentScopeRootTree());
            firstPhaseVisitor.scanNode(assignOperator.getNextSibling());
            new ClassVariableAssignmentVisitor(currentScopeRootTree()).scanNode(assignOperator.getNextSibling());
          }
          if (target.getTokens().size() == 1) {
            addWriteUsage(target.getFirstDescendant(PythonGrammar.NAME));
          }
        }
      }
    }

    private void createFunctionParameters(AstNode functionTree) {
      AstNode parameters = functionTree.getFirstChild(PythonGrammar.TYPEDARGSLIST);
      if (parameters == null) {
        return;
      }
      AstSelect parameterNames = parameters.select()
        .descendants(PythonGrammar.TFPDEF)
        .children(PythonGrammar.NAME);
      for (AstNode parameterName : parameterNames) {
        addWriteUsage(parameterName);
      }
    }

    private void createLambdaParameters(AstNode functionTree) {
      AstNode parameters = functionTree.getFirstChild(PythonGrammar.VARARGSLIST);
      if (parameters == null) {
        return;
      }

      parameters.getChildren(PythonGrammar.NAME).forEach(this::addWriteUsage);
      parameters.getDescendants(PythonGrammar.FPDEF).stream()
        .flatMap(paramDef -> paramDef.getChildren(PythonGrammar.NAME).stream())
        .forEach(this::addWriteUsage);
    }

    private void createLoopVariables(AstNode loopTree) {
      AstNode target = loopTree.getFirstChild(PythonGrammar.EXPRLIST);
      if (target.getTokens().size() == 1) {
        addWriteUsage(target.getFirstDescendant(PythonGrammar.NAME));
      }
    }

    private void createScope(AstNode node, @Nullable Scope parent) {
      scopesByRootTree.put(node, new Scope(parent, node));
    }

    private void addWriteUsage(AstNode nameNode) {
      currentScope().addWriteUsage(nameNode);
    }

    private Scope currentScope() {
      return scopesByRootTree.get(currentScopeRootTree());
    }

  }

  private static class SymbolTablImpl implements SymbolTable {

    private final Map<AstNode, Scope> scopesByRootTree;

    public SymbolTablImpl(Map<AstNode, Scope> scopesByRootTree) {
      this.scopesByRootTree = scopesByRootTree;
    }

    @Override
    public Set<Symbol> symbols(AstNode scopeTree) {
      Scope scope = scopesByRootTree.get(scopeTree);
      return scope == null ? Collections.emptySet() : scope.symbols();
    }

  }

  private static class Scope {

    private final AstNode rootTree;
    private final Scope parent;
    private final Map<String, Symbol> symbolsByName = new HashMap<>();
    private final Set<Symbol> symbols = new HashSet<>();
    private final Set<String> globalNames = new HashSet<>();
    private final Set<String> nonlocalNames = new HashSet<>();

    private Scope(@Nullable Scope parent, AstNode rootTree) {
      this.parent = parent;
      this.rootTree = rootTree;
    }

    private Set<Symbol> symbols() {
      return Collections.unmodifiableSet(symbols);
    }

    public void addWriteUsage(AstNode nameNode) {
      addWriteUsage(nameNode, nameNode.getTokenValue(), null);
    }

    public void addWriteUsage(AstNode nameNode, @Nullable String moduleName) {
     addWriteUsage(nameNode, nameNode.getTokenValue(), moduleName);
    }

    public void addWriteUsage(AstNode nameNode, String symbolName, @Nullable String moduleName) {
      if (!symbolsByName.containsKey(symbolName) && !globalNames.contains(symbolName) && !nonlocalNames.contains(symbolName)) {
        SymbolImpl symbol = new SymbolImpl(symbolName, rootTree);
        symbols.add(symbol);
        symbolsByName.put(symbolName, symbol);
      }
      SymbolImpl symbol = resolve(symbolName);
      if (symbol != null) {
        if (moduleName != null) {
          symbol.moduleName = moduleName;
        }
        symbol.addWriteUsage(nameNode);
      }
    }

    @CheckForNull
    public SymbolImpl resolve(String symbolName) {
      if (nonlocalNames.contains(symbolName)) {
        return resolveNonlocal(symbolName);
      }
      Symbol symbol = symbolsByName.get(symbolName);
      if (parent == null || symbol != null) {
        return (SymbolImpl) symbol;
      }
      if (globalNames.contains(symbolName)) {
        return rootScope().resolve(symbolName);
      }
      return parent.resolve(symbolName);
    }

    private SymbolImpl resolveNonlocal(String symbolName) {
      Scope scope = parent;
      while (scope.parent != null) {
        Symbol symbol = scope.symbolsByName.get(symbolName);
        if (symbol != null) {
          return (SymbolImpl) symbol;
        }
        scope = scope.parent;
      }
      return null;
    }

    private Scope rootScope() {
      Scope scope = this;
      while (scope.parent != null) {
        scope = scope.parent;
      }
      return scope;
    }

    private void addGlobalName(String name) {
      globalNames.add(name);
    }

    private void addNonlocalName(String name) {
      nonlocalNames.add(name);
    }
  }

  private static class SymbolImpl implements Symbol {

    private final String name;
    private String moduleName;
    private final AstNode scopeRootTree;
    private final Set<AstNode> writeUsages = new HashSet<>();
    private final Set<AstNode> readUsages = new HashSet<>();

    private SymbolImpl(String name, AstNode scopeRootTree) {
      this.name = name;
      this.scopeRootTree = scopeRootTree;
      this.moduleName = null;
    }

    @Override
    public String name() {
      return name;
    }

    @Override
    public AstNode scopeTree() {
      return scopeRootTree;
    }

    @Override
    public Set<AstNode> writeUsages() {
      return Collections.unmodifiableSet(writeUsages);
    }

    @Override
    public Set<AstNode> readUsages() {
      return Collections.unmodifiableSet(readUsages);
    }

    @Override
    public String moduleName() {
      return moduleName;
    }

    public void addWriteUsage(AstNode nameNode) {
      writeUsages.add(nameNode);
    }

    public void addReadUsage(AstNode nameNode) {
      readUsages.add(nameNode);
    }
  }

  /**
   * Read usages have to be visited in a second phase.
   * They can't be visited in the same phase as write usages,
   * since a read usage may appear in the syntax tree "before" it's declared (written).
   */
  private class SecondPhaseVisitor extends ScopeVisitor {

    @Override
    public Set<AstNodeType> subscribedKinds() {
      Set<AstNodeType> set = new HashSet<>();
      set.add(PythonGrammar.FUNCDEF);
      set.add(PythonGrammar.LAMBDEF);
      set.add(PythonGrammar.LAMBDEF_NOCOND);
      set.add(PythonGrammar.CLASSDEF);
      set.add(PythonGrammar.ATOM);
      set.add(PythonGrammar.DOTTED_NAME);
      return Collections.unmodifiableSet(set);
    }

    @Override
    public void visitNode(AstNode node) {
      super.visitNode(node);
      if (node.is(PythonGrammar.ATOM, PythonGrammar.DOTTED_NAME)) {
        AstNode nameNode = node.getFirstChild(PythonGrammar.NAME);
        if (nameNode != null) {
          Scope currentScope = scopesByRootTree.get(currentScopeRootTree());
          SymbolImpl symbol = currentScope.resolve(nameNode.getTokenValue());
          if (symbol != null && !symbol.writeUsages.contains(nameNode) && !allReadUsages.contains(nameNode)) {
            symbol.addReadUsage(nameNode);
            allReadUsages.add(nameNode);
          }
        }
      }
    }

  }

  private class ClassVariableAssignmentVisitor extends SecondPhaseVisitor {

    public ClassVariableAssignmentVisitor(AstNode classTree) {
      enterScope(classTree);
    }

  }

  private class Module {
    String name;
    String alias;
  }
}
