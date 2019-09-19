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
import org.sonar.python.api.tree.PyToken;
import java.util.List;
import org.sonar.python.api.tree.PySliceListTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PySliceListTreeImpl extends PyTree implements PySliceListTree {

  private final List<Tree> slices;
  private final List<PyToken> separators;

  public PySliceListTreeImpl(AstNode node, List<Tree> slices, List<PyToken> separators) {
    super(node);
    this.slices = slices;
    this.separators = separators;
  }

  @Override
  public List<Tree> slices() {
    return slices;
  }

  @Override
  public List<PyToken> separators() {
    return separators;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitSliceList(this);
  }

  @Override
  public List<Tree> children() {
    return Stream.of(slices, separators).flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }

  @Override
  public Kind getKind() {
    return Kind.SLICE_LIST;
  }
}
