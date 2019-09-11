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
import java.util.stream.Collectors;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.PythonVisitor;
import org.sonar.python.PythonVisitorContext;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.sslr.ast.AstSelect;

public class SymbolTableBuilderVisitor extends PythonVisitor {

  private Map<AstNode, Scope> scopesByRootTree;
  private Set<AstNode> allReadUsages;
  private Map<String, Module> importedModules = new HashMap<>();
  private Map<AstNode, Symbol> symbolByNode = new HashMap<>();

  public SymbolTable symbolTable() {
    return new SymbolTablImpl(scopesByRootTree, symbolByNode);
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
      set.add(PythonGrammar.ATTRIBUTE_REF);
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

      } else if (node.is(PythonGrammar.ATTRIBUTE_REF)) {
        addSymbolForAttributeRef(node);
      }

    }

    /**
     * Given `myModuleName.f` ATTRIBUTE_REF node, it adds `myModuleName.f` to the symbol table
     * and resolves its qualified name.
     * This is used by rules to easily retrieve the symbol of a "property access" from AstNode of type ATTRIBUTE_REF
     * see {@link SymbolTable#getSymbol(AstNode)}
     */
    private void addSymbolForAttributeRef(AstNode attributeRef) {
      String symbolName = attributeRef.getChildren(PythonGrammar.ATOM, PythonGrammar.NAME).stream()
        .map(AstNode::getTokenValue)
        .collect(Collectors.joining( "." ));
      String propertyName = attributeRef.getLastChild(PythonGrammar.NAME).getTokenValue();
      String namespace = symbolName.replaceAll("\\." + propertyName, "");
      Module module = importedModules.get(namespace);
      if (module != null) {
        SymbolImpl symbol = module.scope.resolve(symbolName);
        if (symbol == null) {
          String qualifiedName = qualifiedName(module.name, propertyName);
          symbol = new SymbolImpl(symbolName, module.scope.rootTree, qualifiedName);
          module.scope.symbols.add(symbol);
          module.scope.symbolsByName.put(symbolName, symbol);
        }
        symbol.addReadUsage(attributeRef);
        symbolByNode.put(attributeRef, symbol);
      }
    }

    /**
     * Adds imported symbols to the symbol table.
     * Keeps track of imported modules and eventually of their aliases.
     * <p>
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
      } else if (node.is(PythonGrammar.IMPORT_FROM)) {
        AstNode dottedName = node.getFirstChild(PythonGrammar.DOTTED_NAME);
        if (dottedName != null) {
          String moduleName = dottedName.getChildren(PythonGrammar.NAME).stream()
            .map(AstNode::getTokenValue)
            .collect(Collectors.joining("."));
          node.getDescendants(PythonGrammar.IMPORT_AS_NAME).forEach(
            importAsName -> {
              // ignore import that contains aliases
              if (importAsName.getChildren(PythonGrammar.NAME).size() == 1) {
                currentScope().addWriteUsage(importAsName.getFirstChild(PythonGrammar.NAME), moduleName);
              }
            });
        }
      }
    }

    private void addImportedSymbols(AstNode moduleNameNode, @Nullable AstNode aliasNode) {
      String moduleName = moduleNameNode.getChildren(PythonGrammar.NAME).stream()
        .map(AstNode::getTokenValue)
        .collect(Collectors.joining("."));
      if (aliasNode != null) {
        currentScope().addWriteUsage(aliasNode);
        String alias = aliasNode.getTokenValue();
        importedModules.put(alias, new Module(moduleName, currentScope(), alias));
      } else {
        currentScope().addWriteUsage(moduleNameNode);
        importedModules.put(moduleName, new Module(moduleName, currentScope(), null));
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
    private final Map<AstNode, Symbol> symbolByNode;

    public SymbolTablImpl(Map<AstNode, Scope> scopesByRootTree, Map<AstNode, Symbol> symbolByNode) {
      this.scopesByRootTree = scopesByRootTree;
      this.symbolByNode = Collections.unmodifiableMap(symbolByNode);
    }

    @Override
    public Set<Symbol> symbols(AstNode scopeTree) {
      Scope scope = scopesByRootTree.get(scopeTree);
      return scope == null ? Collections.emptySet() : scope.symbols();
    }

    @CheckForNull
    @Override
    public Symbol getSymbol(AstNode node) {
      return symbolByNode.get(node);
    }

    @Override
    @CheckForNull
    public Symbol getSymbol(PyExpressionTree expression) {
      AstNode astNode = expression.astNode();
      // strongly typed ast doesn't have ATOM node, we check for the parent in case of a PyTreeNode
      if (astNode != null && astNode.is(PythonGrammar.NAME) && astNode.getParent().is(PythonGrammar.ATOM)) {
        return getSymbol(astNode.getParent());
      }
      if (astNode != null && astNode.is(PythonGrammar.NAME) && astNode.getParent().is(PythonGrammar.ATTRIBUTE_REF) && astNode.getParent().getLastChild(PythonGrammar.NAME) == astNode) {
        return getSymbol(astNode.getParent());
      }
      return getSymbol(astNode);
    }

  }

  private class Scope {

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
      addWriteUsage(nameNode, null);
    }

    public void addWriteUsage(AstNode nameNode, @Nullable String moduleName) {
      String symbolName = nameNode.getTokenValue();
      if (!symbolsByName.containsKey(symbolName) && !globalNames.contains(symbolName) && !nonlocalNames.contains(symbolName)) {
        SymbolImpl symbol = new SymbolImpl(symbolName, rootTree, qualifiedName(moduleName, symbolName));
        symbols.add(symbol);
        symbolsByName.put(symbolName, symbol);
        symbolByNode.put(nameNode, symbol);
      }
      SymbolImpl symbol = resolve(symbolName);
      if (symbol != null) {
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

  @CheckForNull
  private static String qualifiedName(@Nullable String moduleName, String symbolName) {
    return moduleName == null ? null : moduleName + "." + symbolName;
  }

  private static class SymbolImpl implements Symbol {

    private final String name;
    private final String qualifiedName;
    private final AstNode scopeRootTree;
    private final Set<AstNode> writeUsages = new HashSet<>();
    private final Set<AstNode> readUsages = new HashSet<>();

    private SymbolImpl(String name, AstNode scopeRootTree, @Nullable String qualifiedName) {
      this.name = name;
      this.scopeRootTree = scopeRootTree;
      this.qualifiedName = qualifiedName;
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
    public String qualifiedName() {
      if (qualifiedName == null) {
        return name;
      }
      return qualifiedName;
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
      set.add(PythonGrammar.CALL_EXPR);
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
            symbolByNode.put(node, symbol);
          }
        }
      } else if (node.is(PythonGrammar.CALL_EXPR)) {
        addSymbolForCallExpression(node);
      }
    }

    /**
     * This is used by rules to easily retrieve the symbol of a function from AstNode of type CALL_EXPR
     * see {@link SymbolTable#getSymbol(AstNode)}
     */
    private void addSymbolForCallExpression(AstNode node) {
      Scope currentScope = scopesByRootTree.get(currentScopeRootTree());
      String symbolName = "";
      AstNode firstChild = node.getFirstChild();
      if (firstChild.is(PythonGrammar.ATTRIBUTE_REF)) {
        symbolName = firstChild.getChildren(PythonGrammar.ATOM, PythonGrammar.NAME).stream()
          .map(AstNode::getTokenValue)
          .collect(Collectors.joining( "." ));
      } else if(firstChild.is(PythonGrammar.ATOM)) {
        symbolName = firstChild.getTokenValue();
      }
      SymbolImpl symbol = currentScope.resolve(symbolName);
      if (symbol != null) {
        if (firstChild.is(PythonGrammar.ATOM)) {
          symbol.addReadUsage(node);
        }
        symbolByNode.put(node, symbol);
      }
    }

  }

  private class ClassVariableAssignmentVisitor extends SecondPhaseVisitor {

    public ClassVariableAssignmentVisitor(AstNode classTree) {
      enterScope(classTree);
    }

  }

  private static class Module {
    final String name;
    final String alias;
    final Scope scope;

    Module(String name, Scope scope, @Nullable String alias) {
      this.name = name;
      this.alias = alias;
      this.scope = scope;
    }
  }
}
