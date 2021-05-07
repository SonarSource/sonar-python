/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.python.index;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.semantic.AmbiguousSymbolImpl;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.semantic.SymbolImpl;
import org.sonar.python.types.TypeShed;

public class SymbolBuilder {
  private Collection<Summary> summaries = new HashSet<>();
  private final Map<String, Symbol> existingSymbols;
  private final ProjectSummary projectSummary;
  @Nullable
  private String fullyQualifiedName;
  private String alias;

  public SymbolBuilder(Map<String, Symbol> existingSymbols, ProjectSummary projectSummary) {
    this.existingSymbols = existingSymbols;
    this.projectSummary = projectSummary;
  }

  public SymbolBuilder fromSummaries(Collection<Summary> summaries) {
    this.summaries = summaries;
    return this;
  }

  public SymbolBuilder fromFullyQualifiedName(@Nullable String fullyQualifiedName) {
    this.fullyQualifiedName = fullyQualifiedName;
    for (ModuleSummary module : projectSummary.modules().values()) {
      summaries = module.summariesWithFQN(fullyQualifiedName);
      if (!summaries.isEmpty()) {
        return this;
      }
    }
    return this;
  }

  public SymbolBuilder havingAlias(String alias) {
    this.alias = alias;
    return this;
  }

  @CheckForNull
  public Symbol build() {
    Set<Symbol> symbols = new HashSet<>();
    for (Summary summary : summaries) {
      String name = alias != null ? alias : summary.name();
      if (summary instanceof FunctionSummary) {
        symbols.add(new FunctionSymbolImpl(name, ((FunctionSummary) summary)));
      } else if (summary instanceof ClassSummary) {
        symbols.add(classSymbol(name, ((ClassSummary) summary)));
      } else if (summary instanceof VariableSummary) {
        symbols.add(new SymbolImpl(name, summary.fullyQualifiedName(), ((VariableSummary) summary).annotatedType()));
      }
    }
    if (symbols.isEmpty()) {
      return fullyQualifiedName != null ? TypeShed.symbolWithFQN(fullyQualifiedName) : null;
    }
    if (symbols.size() == 1) {
      return symbols.iterator().next();
    }
    return AmbiguousSymbolImpl.create(symbols);
  }

  private Symbol classSymbol(String name, ClassSummary classSummary) {
    ClassSymbolImpl classSymbol = new ClassSymbolImpl(name, classSummary.fullyQualifiedName(), classSummary.definitionLocation(),
      classSummary.hasDecorators(), classSummary.hasMetaClass(), classSummary.metaclassFQN(), classSummary.supportsGenerics());
    if (classSummary.hasSuperClassWithoutSymbol()) {
      classSymbol.setHasSuperClassWithoutSymbol();
    }
    addSuperClasses(classSummary, classSymbol);
    addMembers(classSummary, classSymbol);
    return classSymbol;
  }

  private void addSuperClasses(ClassSummary classSummary, ClassSymbolImpl classSymbol) {
    for (String superClass : classSummary.superClasses()) {
      Symbol superClassSymbol = existingSymbols.get(superClass);
      if (superClassSymbol == null) {
        superClassSymbol = new SymbolBuilder(existingSymbols, projectSummary)
          .fromFullyQualifiedName(superClass)
          .build();
      }
      if (superClassSymbol != null) {
        classSymbol.addSuperClass(superClassSymbol);
      } else {
        classSymbol.addSuperClass(new SymbolImpl(fullyQualifiedName, fullyQualifiedName));
      }
    }
  }

  private void addMembers(ClassSummary classSummary, ClassSymbolImpl classSymbol) {
    Set<Symbol> members = new HashSet<>();
    for (Summary member : classSummary.members()) {
      Symbol memberSymbol = new SymbolBuilder(existingSymbols, projectSummary)
        .fromSummaries(Collections.singleton(member))
        .build();
      members.add(memberSymbol);
    }
    classSymbol.addMembers(members);
  }
}
