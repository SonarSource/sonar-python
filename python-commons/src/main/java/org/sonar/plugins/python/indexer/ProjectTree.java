/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
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

import java.util.List;
import java.util.stream.Stream;
import javax.annotation.Nullable;

public sealed interface ProjectTree {
  String name();

  @Nullable
  ProjectTreeFolder parent();

  default Stream<ProjectTreeFolder> parents() {
    return Stream.iterate(parent(), node -> node.parent() != null, ProjectTreeFolder::parent);
  }

  Stream<ProjectTreeFolder> allFolders();

  final class ProjectTreeFile implements ProjectTree {
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

  final class ProjectTreeFolder implements ProjectTree {
    private final String name;
    private final List<ProjectTree> children;
    private ProjectTreeFolder parent;

    public ProjectTreeFolder(String name, List<ProjectTree> children) {
      this.name = name;
      this.children = List.copyOf(children);
      this.parent = null;

      for (ProjectTree child : this.children) {
        if (child instanceof ProjectTreeFile file) {
          file.setParent(this);
        } else if (child instanceof ProjectTreeFolder folder) {
          folder.setParent(this);
        }
      }
    }

    @Override
    public String name() {
      return name;
    }

    public List<ProjectTree> children() {
      return children;
    }

    @Override
    @Nullable
    public ProjectTreeFolder parent() {
      return parent;
    }

    @Override
    public Stream<ProjectTreeFolder> allFolders() {
      return Stream.concat(Stream.of(this), children.stream().flatMap(ProjectTree::allFolders));
    }

    void setParent(ProjectTreeFolder parent) {
      this.parent = parent;
    }
  }
}

