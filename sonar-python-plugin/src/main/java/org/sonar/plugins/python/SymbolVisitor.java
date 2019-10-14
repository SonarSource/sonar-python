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
package org.sonar.plugins.python;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import org.sonar.api.batch.sensor.symbol.NewSymbol;
import org.sonar.api.batch.sensor.symbol.NewSymbolTable;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.semantic.Symbol;
import org.sonar.python.semantic.Usage;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;

public class SymbolVisitor extends BaseTreeVisitor {

  private final NewSymbolTable newSymbolTable;

  public SymbolVisitor(NewSymbolTable newSymbolTable) {
    this.newSymbolTable = newSymbolTable;
  }

  @Override
  public void visitClassDef(ClassDef classDef) {
    classDef.classFields().forEach(this::handleSymbol);
    classDef.instanceFields().forEach(this::handleSymbol);
    super.visitClassDef(classDef);
  }

  @Override
  public void visitFunctionDef(FunctionDef functionDef) {
    functionDef.localVariables().forEach(this::handleSymbol);
    super.visitFunctionDef(functionDef);
  }

  @Override
  public void visitLambda(LambdaExpression lambdaExpression) {
    lambdaExpression.localVariables().forEach(this::handleSymbol);
    super.visitLambda(lambdaExpression);
  }

  @Override
  public void visitFileInput(FileInput fileInput) {
    fileInput.globalVariables().forEach(this::handleSymbol);
    super.visitFileInput(fileInput);
    newSymbolTable.save();
  }

  private void handleSymbol(Symbol symbol) {
    List<Usage> usages = new ArrayList<>(symbol.usages());
    usages.sort(Comparator.comparingInt(u -> u.tree().firstToken().line()));
    Tree firstUsageTree = usages.get(0).tree();
    NewSymbol newSymbol = newSymbolTable.newSymbol(firstUsageTree.firstToken().line(), firstUsageTree.firstToken().column(),
      firstUsageTree.lastToken().line(), firstUsageTree.lastToken().column() + firstUsageTree.lastToken().value().length());
    for (int i = 1; i < usages.size(); i++) {
      Tree usageTree = usages.get(i).tree();
      newSymbol.newReference(usageTree.firstToken().line(), usageTree.firstToken().column(),
        usageTree.lastToken().line(), usageTree.lastToken().column() + usageTree.lastToken().value().length());
    }
  }
}
