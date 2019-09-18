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
package org.sonar.python.tree;

import com.sonar.sslr.api.AstNode;
import org.sonar.python.api.tree.PyToken;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.tree.PyAliasedNameTree;
import org.sonar.python.api.tree.PyDottedNameTree;
import org.sonar.python.api.tree.PyImportFromTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyImportFromTreeImpl extends PyTree implements PyImportFromTree {
  private final PyToken fromKeyword;
  private final List<PyToken> dottedPrefixForModule;
  private final PyDottedNameTree moduleName;
  private final PyToken importKeyword;
  private final List<PyAliasedNameTree> aliasedImportNames;
  private final boolean isWildcardImport;
  private final PyToken wildcard;

  public PyImportFromTreeImpl(AstNode astNode, PyToken fromKeyword, @Nullable List<PyToken> dottedPrefixForModule,
                              @Nullable PyDottedNameTree moduleName, PyToken importKeyword,
                              @Nullable List<PyAliasedNameTree> aliasedImportNames, boolean isWildcardImport) {
    super(astNode);
    this.fromKeyword = fromKeyword;
    this.dottedPrefixForModule = dottedPrefixForModule;
    this.moduleName = moduleName;
    this.importKeyword = importKeyword;
    this.aliasedImportNames = aliasedImportNames == null ? Collections.emptyList() : aliasedImportNames;
    this.isWildcardImport = isWildcardImport;
    this.wildcard = isWildcardImport ? new PyTokenImpl(astNode.getFirstChild(PythonPunctuator.MUL).getToken()) : null;
  }

  @Override
  public PyToken fromKeyword() {
    return fromKeyword;
  }

  @CheckForNull
  @Override
  public PyDottedNameTree module() {
    return moduleName;
  }

  @Override
  public PyToken importKeyword() {
    return importKeyword;
  }

  @CheckForNull
  @Override
  public List<PyToken> dottedPrefixForModule() {
    return dottedPrefixForModule;
  }

  @Override
  public List<PyAliasedNameTree> importedNames() {
    return aliasedImportNames;
  }

  @Override
  public boolean isWildcardImport() {
    return isWildcardImport;
  }

  @CheckForNull
  @Override
  public PyToken wildcard() {
    return wildcard;
  }

  @Override
  public Kind getKind() {
    return Kind.IMPORT_FROM;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitImportFrom(this);
  }

  @Override
  public List<Tree> children() {
    return Stream.of(aliasedImportNames, Collections.singletonList(moduleName))
      .flatMap(List::stream).collect(Collectors.toList());
  }
}
