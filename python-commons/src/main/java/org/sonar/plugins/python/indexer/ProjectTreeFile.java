/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python.indexer;

import java.util.stream.Stream;
import javax.annotation.Nullable;

public final class ProjectTreeFile implements ProjectTree {
  private final String name;
  private ProjectTreeFolder parent;

  public ProjectTreeFile(String name) {
    this.name = name;
    this.parent = null;
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  @Nullable
  public ProjectTreeFolder parent() {
    return parent;
  }

  @Override
  public Stream<ProjectTreeFolder> allFolders() {
    return Stream.empty();
  }

  void setParent(ProjectTreeFolder parent) {
    this.parent = parent;
  }
}
