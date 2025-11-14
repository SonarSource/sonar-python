/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.plugins.python.PythonInputFile;

public class ProjectTreeBuilder {
  
  private static final Logger LOG = LoggerFactory.getLogger(ProjectTreeBuilder.class);
  
  private final Map<ProjectPath, FolderNode> folderNodes = new HashMap<>();
  private final FolderNode rootNode = new FolderNode("/");

  public ProjectTree build(List<PythonInputFile> inputFiles) {
    for (PythonInputFile inputFile : inputFiles) {
      addFileToTree(inputFile);
    }
    
    return buildImmutableFolder(rootNode);
  }
  
  private void addFileToTree(PythonInputFile inputFile) {
    ProjectPath relativePath = extractRelativePath(inputFile);
    if (relativePath == null) {
      return;
    }
    
    getOrCreateFolder(relativePath);
  }

  private FolderNode getOrCreateFolder(ProjectPath path) {
    if (path.isEmpty()) {
      return rootNode;
    }

    if (folderNodes.containsKey(path)) {
      return folderNodes.get(path);
    }

    String fileName = path.fileName();
    if (fileName == null) {
      fileName = "";
    }
    var node = new FolderNode(fileName);
    folderNodes.put(path, node);

    FolderNode parent = getOrCreateFolder(path.parent());
    parent.children.add(node);
    return node;
  }

  private static ProjectTree buildImmutableFolder(FolderNode node) {
    ArrayList<ProjectTree> children = new ArrayList<>();
    for (FolderNode childNode : node.children) {
      children.add(buildImmutableFolder(childNode));
    }

    if (children.isEmpty()) {
      return new ProjectTree.ProjectTreeFile(node.name);
    } else {
      return new ProjectTree.ProjectTreeFolder(node.name, List.copyOf(children));
    }
  }

  
  @CheckForNull
  private static ProjectPath extractRelativePath(PythonInputFile inputFile) {
    URI uri = inputFile.wrappedFile().uri();

    if (!"file".equalsIgnoreCase(uri.getScheme())) {
      LOG.debug("Skipping non-file URI: {}", uri);
      return null;
    }
    return ProjectPath.parse(uri.getSchemeSpecificPart());
  }


  private record ProjectPath(List<String> segments) {
    static ProjectPath parse(@Nullable String pathString) {
      if (pathString == null || pathString.isEmpty()) {
        return new ProjectPath(List.of());
      }

      String[] parts = pathString.split("[/\\\\]+");
      List<String> nonEmptySegments = Arrays.stream(parts).filter(part -> !part.isEmpty()).toList();
      return new ProjectPath(nonEmptySegments);
    }

    boolean isEmpty() {
      return segments.isEmpty();
    }

    @CheckForNull
    String fileName() {
      return isEmpty() ? null : segments.get(segments.size() - 1);
    }

    ProjectPath parent() {
      if (segments.size() <= 1) {
        return new ProjectPath(List.of());
      }
      return new ProjectPath(segments.subList(0, segments.size() - 1));
    }
  }

  private static class FolderNode {
    final String name;
    final List<FolderNode> children = new ArrayList<>();
    
    FolderNode(String name) {
      this.name = name;
    }
  }
}

