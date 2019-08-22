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
import com.sonar.sslr.api.Token;
import java.util.List;
import javax.annotation.CheckForNull;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.tree.PyAliasedNameTree;
import org.sonar.python.api.tree.PyDottedNameTree;
import org.sonar.python.api.tree.PyImportFromTree;
import org.sonar.python.api.tree.PyTreeVisitor;

public class PyImportFromTreeImpl extends PyTree implements PyImportFromTree {
  private final Token fromKeyword;
  private final List<Token> dottedPrefixForModule;
  private final PyDottedNameTree moduleName;
  private final Token importKeyword;
  private final List<PyAliasedNameTree> aliasedImportNames;
  private final boolean isWildcardImport;
  private final Token wildcard;

  public PyImportFromTreeImpl(AstNode astNode, Token fromKeyword, List<Token> dottedPrefixForModule, PyDottedNameTree moduleName, Token importKeyword, List<PyAliasedNameTree> aliasedImportNames, boolean isWildcardImport) {
    super(astNode);
    this.fromKeyword = fromKeyword;
    this.dottedPrefixForModule = dottedPrefixForModule;
    this.moduleName = moduleName;
    this.importKeyword = importKeyword;
    this.aliasedImportNames = aliasedImportNames;
    this.isWildcardImport = isWildcardImport;
    this.wildcard = isWildcardImport ? astNode.getFirstChild(PythonPunctuator.MUL).getToken() : null;
  }

  @Override
  public Token fromKeyword() {
    return fromKeyword;
  }

  @CheckForNull
  @Override
  public PyDottedNameTree module() {
    return moduleName;
  }

  @Override
  public Token importKeyword() {
    return importKeyword;
  }

  @CheckForNull
  @Override
  public List<Token> dottedPrefixForModule() {
    return dottedPrefixForModule;
  }

  @CheckForNull
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
  public Token wildcard() {
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
}
