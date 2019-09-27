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
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.tree.AliasedName;
import org.sonar.python.api.tree.DottedName;
import org.sonar.python.api.tree.ImportFrom;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.TreeVisitor;
import org.sonar.python.api.tree.Tree;

public class ImportFromImpl extends PyTree implements ImportFrom {
  private final Token fromKeyword;
  private final List<Token> dottedPrefixForModule;
  private final DottedName moduleName;
  private final Token importKeyword;
  private final List<AliasedName> aliasedImportNames;
  private final boolean isWildcardImport;
  private final Token wildcard;

  public ImportFromImpl(AstNode astNode, Token fromKeyword, @Nullable List<Token> dottedPrefixForModule,
                        @Nullable DottedName moduleName, Token importKeyword,
                        @Nullable List<AliasedName> aliasedImportNames, boolean isWildcardImport) {
    super(astNode);
    this.fromKeyword = fromKeyword;
    this.dottedPrefixForModule = dottedPrefixForModule;
    this.moduleName = moduleName;
    this.importKeyword = importKeyword;
    this.aliasedImportNames = aliasedImportNames == null ? Collections.emptyList() : aliasedImportNames;
    this.isWildcardImport = isWildcardImport;
    this.wildcard = isWildcardImport ? new TokenImpl(astNode.getFirstChild(PythonPunctuator.MUL).getToken()) : null;
  }

  @Override
  public Token fromKeyword() {
    return fromKeyword;
  }

  @CheckForNull
  @Override
  public DottedName module() {
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

  @Override
  public List<AliasedName> importedNames() {
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
  public void accept(TreeVisitor visitor) {
    visitor.visitImportFrom(this);
  }

  @Override
  public List<Tree> children() {
    return Stream.of(Collections.singletonList(importKeyword), aliasedImportNames, Collections.singletonList(fromKeyword),
      dottedPrefixForModule, Arrays.asList(moduleName, wildcard)).flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
