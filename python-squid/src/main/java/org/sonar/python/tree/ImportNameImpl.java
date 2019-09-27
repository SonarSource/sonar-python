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
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.Token;
import java.util.Collections;
import java.util.List;
import org.sonar.python.api.tree.AliasedName;
import org.sonar.python.api.tree.ImportName;
import org.sonar.python.api.tree.TreeVisitor;
import org.sonar.python.api.tree.Tree;

public class ImportNameImpl extends PyTree implements ImportName {

  private final Token importKeyword;
  private final List<AliasedName> aliasedNames;
  private final Token separator;

  public ImportNameImpl(AstNode astNode, Token importKeyword, List<AliasedName> aliasedNames, @Nullable Token separator) {
    super(astNode);
    this.importKeyword = importKeyword;
    this.aliasedNames = aliasedNames;
    this.separator = separator;
  }

  @Override
  public Token importKeyword() {
    return importKeyword;
  }

  @Override
  public List<AliasedName> modules() {
    return aliasedNames;
  }

  @CheckForNull
  @Override
  public Token separator() {
    return separator;
  }

  @Override
  public Kind getKind() {
    return Kind.IMPORT_NAME;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitImportName(this);
  }

  @Override
  public List<Tree> children() {
    return Stream.of(Collections.singletonList(importKeyword), aliasedNames, Collections.singletonList(separator)).flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
