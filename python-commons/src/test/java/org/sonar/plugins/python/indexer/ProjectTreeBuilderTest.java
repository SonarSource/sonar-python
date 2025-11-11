/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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

import java.net.URI;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.plugins.python.PythonInputFile;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ProjectTreeBuilderTest {

  @Test
  void test_empty_file_list() {
    ProjectTreeBuilder builder = new ProjectTreeBuilder();

    ProjectTree tree = builder.build(List.of());

    assertThat(tree).isInstanceOf(ProjectTree.ProjectTreeFile.class);
    assertThat(tree.name()).isEqualTo("/");
  }

  @ParameterizedTest
  @MethodSource("provideRootPathVariations")
  void test_windows_path(String rootPath) {
    ProjectTreeBuilder builder = new ProjectTreeBuilder();
    PythonInputFile rootFile = createMockInputFileWithScheme("file:" + rootPath);

    ProjectTree tree = builder.build(List.of(rootFile));
    assertThat(tree).isInstanceOf(ProjectTree.ProjectTreeFolder.class);
    ProjectTree.ProjectTreeFolder root = (ProjectTree.ProjectTreeFolder) tree;
    assertThat(root.name()).isEqualTo("/");
    assertThat(root.children()).hasSize(1);

    ProjectTree driveLetterFolder = root.children().get(0);
    assertThat(driveLetterFolder.name()).isEqualTo("c:");
    assertThat(driveLetterFolder).isInstanceOf(ProjectTree.ProjectTreeFile.class);
  }

  static Stream<String> provideRootPathVariations() {
    return Stream.of(
      "/c:/",
      "/c:");
  }

  @Test
  void test_single_file() {
    ProjectTreeBuilder builder = new ProjectTreeBuilder();
    PythonInputFile inputFile = createMockInputFile("/project/file.py");

    ProjectTree tree = builder.build(List.of(inputFile));

    assertThat(tree).isInstanceOf(ProjectTree.ProjectTreeFolder.class);
    ProjectTree.ProjectTreeFolder root = (ProjectTree.ProjectTreeFolder) tree;
    assertThat(root.name()).isEqualTo("/");
    assertThat(root.children()).hasSize(1);

    ProjectTree projectFolder = root.children().get(0);
    assertThat(projectFolder.name()).isEqualTo("project");
    assertThat(projectFolder).isInstanceOf(ProjectTree.ProjectTreeFolder.class);

    ProjectTree.ProjectTreeFolder projectFolderCast = (ProjectTree.ProjectTreeFolder) projectFolder;
    assertThat(projectFolderCast.children()).hasSize(1);
    assertThat(projectFolderCast.children().get(0).name()).isEqualTo("file.py");
  }

  @Test
  void test_multiple_files_same_folder() {
    ProjectTreeBuilder builder = new ProjectTreeBuilder();
    PythonInputFile file1 = createMockInputFile("/project/file1.py");
    PythonInputFile file2 = createMockInputFile("/project/file2.py");
    PythonInputFile file3 = createMockInputFile("/project/file3.py");

    ProjectTree tree = builder.build(List.of(file1, file2, file3));

    ProjectTree.ProjectTreeFolder root = (ProjectTree.ProjectTreeFolder) tree;
    ProjectTree.ProjectTreeFolder project = (ProjectTree.ProjectTreeFolder) root.children().get(0);

    assertThat(project.children()).hasSize(3);
    List<String> fileNames = project.children().stream().map(ProjectTree::name).toList();
    assertThat(fileNames).containsExactlyInAnyOrder("file1.py", "file2.py", "file3.py");
  }

  @Test
  void test_non_file_uri_is_skipped() {
    ProjectTreeBuilder builder = new ProjectTreeBuilder();
    PythonInputFile httpFile = createMockInputFileWithScheme("http://example.com/file.py");
    PythonInputFile normalFile = createMockInputFile("/project/file.py");

    ProjectTree tree = builder.build(List.of(httpFile, normalFile));

    ProjectTree.ProjectTreeFolder root = (ProjectTree.ProjectTreeFolder) tree;
    ProjectTree.ProjectTreeFolder project = (ProjectTree.ProjectTreeFolder) root.children().get(0);

    assertThat(project.children()).hasSize(1);
    assertThat(project.children().get(0).name()).isEqualTo("file.py");
  }

  @Test
  void test_relative_paths() {
    ProjectTreeBuilder builder = new ProjectTreeBuilder();

    // File at project/foo/file.py (relative to working directory)
    PythonInputFile relativeFile = mock(PythonInputFile.class);
    InputFile wrappedFile = mock(InputFile.class);
    when(relativeFile.wrappedFile()).thenReturn(wrappedFile);
    // Use a relative URI, not absolute
    when(wrappedFile.uri()).thenReturn(URI.create("file:project/foo/file.py"));

    ProjectTree tree = builder.build(List.of(relativeFile));

    assertThat(tree).isInstanceOf(ProjectTree.ProjectTreeFolder.class);
    ProjectTree.ProjectTreeFolder root = (ProjectTree.ProjectTreeFolder) tree;
    assertThat(root.name()).isEqualTo("/");
    assertThat(root.children()).hasSize(1);

    ProjectTree projectFolder = root.children().get(0);
    assertThat(projectFolder.name()).isEqualTo("project");
    assertThat(projectFolder).isInstanceOf(ProjectTree.ProjectTreeFolder.class);

    ProjectTree.ProjectTreeFolder projectFolderCast = (ProjectTree.ProjectTreeFolder) projectFolder;
    assertThat(projectFolderCast.children()).hasSize(1);

    ProjectTree fooFolder = projectFolderCast.children().get(0);
    assertThat(fooFolder.name()).isEqualTo("foo");
    assertThat(fooFolder).isInstanceOf(ProjectTree.ProjectTreeFolder.class);

    ProjectTree.ProjectTreeFolder fooFolderCast = (ProjectTree.ProjectTreeFolder) fooFolder;
    assertThat(fooFolderCast.children()).hasSize(1);
    assertThat(fooFolderCast.children().get(0).name()).isEqualTo("file.py");
  }

  private PythonInputFile createMockInputFile(String path) {
    PythonInputFile inputFile = mock(PythonInputFile.class);
    InputFile wrappedFile = mock(InputFile.class);
    when(inputFile.wrappedFile()).thenReturn(wrappedFile);
    when(wrappedFile.uri()).thenReturn(URI.create("file:" + path));
    return inputFile;
  }

  private PythonInputFile createMockInputFileWithScheme(String uriString) {
    PythonInputFile inputFile = mock(PythonInputFile.class);
    InputFile wrappedFile = mock(InputFile.class);
    when(inputFile.wrappedFile()).thenReturn(wrappedFile);
    when(wrappedFile.uri()).thenReturn(URI.create(uriString));
    return inputFile;
  }

  @ParameterizedTest
  @MethodSource("providePathVariations")
  void test_path_parsing_normalizes_all_path_formats(String pathInput) {
    ProjectTreeBuilder builder = new ProjectTreeBuilder();
    PythonInputFile file = createMockInputFileWithPath(pathInput);

    ProjectTree tree = builder.build(List.of(file));

    // Expecting project -> src -> main -> file.py
    ProjectTree.ProjectTreeFolder root = (ProjectTree.ProjectTreeFolder) tree;
    assertThat(root.children()).hasSize(1);

    ProjectTree.ProjectTreeFolder project = (ProjectTree.ProjectTreeFolder) root.children().get(0);
    assertThat(project.name()).isEqualTo("project");
    assertThat(project.children()).hasSize(1);

    ProjectTree.ProjectTreeFolder src = (ProjectTree.ProjectTreeFolder) project.children().get(0);
    assertThat(src.name()).isEqualTo("src");
    assertThat(src.children()).hasSize(1);

    ProjectTree.ProjectTreeFolder main = (ProjectTree.ProjectTreeFolder) src.children().get(0);
    assertThat(main.name()).isEqualTo("main");
    assertThat(main.children()).hasSize(1);

    assertThat(main.children().get(0).name()).isEqualTo("file.py");
  }

  static Stream<String> providePathVariations() {
    return Stream.of(
      "project/src/main/file.py",
      "project\\src\\main\\file.py",
      "project/src\\main/file.py",
      "project//src///main/file.py",
      "/project/src/main/file.py",
      "//project/src/main/file.py",
      "//project///src//main////file.py",
      "\\project\\src\\main\\file.py",
      "/project\\src/main\\file.py",
      "project/src/main/file.py/",
      "project\\src\\main\\file.py\\",
      "project/src/main/file.py///",
      "/project/src/main/file.py/",
      "//project///src\\\\main//file.py///");
  }

  private PythonInputFile createMockInputFileWithPath(String path) {
    PythonInputFile inputFile = mock(PythonInputFile.class);
    InputFile wrappedFile = mock(InputFile.class);
    when(inputFile.wrappedFile()).thenReturn(wrappedFile);
    // Create a URI that will have the path as the scheme-specific part
    URI uri = mock(URI.class);
    when(uri.getScheme()).thenReturn("file");
    when(uri.getSchemeSpecificPart()).thenReturn(path);
    when(wrappedFile.uri()).thenReturn(uri);
    return inputFile;
  }
}
