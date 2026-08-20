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

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.sonar.api.batch.fs.FileSystem;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.config.Configuration;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PackageRootResolverTest {

  @TempDir
  Path tempDir;

  // ─── Helpers ───────────────────────────────────────────────────────────────

  private FileSystem mockFileSystem(File baseDir) {
    FileSystem fs = mock(FileSystem.class);
    when(fs.baseDir()).thenReturn(baseDir);
    return fs;
  }

  private InputFile mockInputFile(File file) {
    InputFile inputFile = mock(InputFile.class);
    when(inputFile.file()).thenReturn(file);
    return inputFile;
  }

  private Configuration noSonarSources() {
    Configuration config = mock(Configuration.class);
    when(config.getStringArray("sonar.sources")).thenReturn(new String[0]);
    return config;
  }

  private Configuration sonarSources(String... values) {
    Configuration config = mock(Configuration.class);
    when(config.getStringArray("sonar.sources")).thenReturn(values);
    return config;
  }

  // ─── resolve() — no build config files → LEGACY_INIT_PY ──────────────────

  @Test
  void resolve_noBuildConfigFiles_noPythonFiles_returnsBaseDirAsRoot() {
    File baseDir = tempDir.toFile();

    PackageResolutionResult result = PackageRootResolver.resolve(mockFileSystem(baseDir), noSonarSources(), List.of());

    assertThat(result.roots()).containsExactly(baseDir.getAbsolutePath());
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.LEGACY_INIT_PY);
  }

  @Test
  void resolve_inaccessibleBaseDir_fallsBackToLegacyResolution() {
    File missingBaseDir = tempDir.resolve("missing").toFile();

    PackageResolutionResult result = PackageRootResolver.resolve(mockFileSystem(missingBaseDir), noSonarSources(), List.of());

    assertThat(result.roots()).containsExactly(missingBaseDir.getAbsolutePath());
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.LEGACY_INIT_PY);
  }

  @Test
  void resolve_noBuildConfigFiles_filesWithNoInitPy_returnsParentDirsAsRoots() throws IOException {
    // Files with no __init__.py in their parent: root is the parent dir itself
    File baseDir = tempDir.toFile();
    File pkgDir = tempDir.resolve("mypkg").toFile();
    pkgDir.mkdir();
    File pyFile = new File(pkgDir, "module.py");
    pyFile.createNewFile();

    PackageResolutionResult result = PackageRootResolver.resolve(
      mockFileSystem(baseDir), noSonarSources(), List.of(mockInputFile(pyFile)));

    // mypkg/ has no __init__.py, so root is mypkg/ itself
    assertThat(result.roots()).containsExactly(pkgDir.getAbsolutePath());
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.LEGACY_INIT_PY);
  }

  @Test
  void resolve_noBuildConfigFiles_filesWithInitPy_walksUpToFindRoot() throws IOException {
    // pkg/__init__.py exists → root walks up to baseDir
    File baseDir = tempDir.toFile();
    File pkgDir = tempDir.resolve("mypkg").toFile();
    pkgDir.mkdir();
    new File(pkgDir, "__init__.py").createNewFile();
    File pyFile = new File(pkgDir, "module.py");
    pyFile.createNewFile();

    PackageResolutionResult result = PackageRootResolver.resolve(
      mockFileSystem(baseDir), noSonarSources(), List.of(mockInputFile(pyFile)));

    // mypkg/ has __init__.py → walk up → baseDir has no __init__.py → root = baseDir
    assertThat(result.roots()).containsExactly(baseDir.getAbsolutePath());
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.LEGACY_INIT_PY);
  }

  @Test
  void resolve_noBuildConfigFiles_srcFolderPresent_stillComputesLegacyRoots() throws IOException {
    // Conventional folders are irrelevant when no build config files exist; roots come from __init__.py walk
    Files.createDirectory(tempDir.resolve("src"));
    File baseDir = tempDir.toFile();
    File pkgDir = tempDir.resolve("mypkg").toFile();
    pkgDir.mkdir();
    new File(pkgDir, "__init__.py").createNewFile();
    File pyFile = new File(pkgDir, "module.py");
    pyFile.createNewFile();

    PackageResolutionResult result = PackageRootResolver.resolve(
      mockFileSystem(baseDir), noSonarSources(), List.of(mockInputFile(pyFile)));

    assertThat(result.roots()).containsExactly(baseDir.getAbsolutePath());
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.LEGACY_INIT_PY);
  }

  @Test
  void resolve_noBuildConfigFiles_sonarSourcesIgnored() throws IOException {
    // sonar.sources is also irrelevant when no build config files exist
    File baseDir = tempDir.toFile();
    File pkgDir = tempDir.resolve("mypkg").toFile();
    pkgDir.mkdir();
    new File(pkgDir, "__init__.py").createNewFile();
    File pyFile = new File(pkgDir, "module.py");
    pyFile.createNewFile();

    PackageResolutionResult result = PackageRootResolver.resolve(
      mockFileSystem(baseDir), sonarSources("."), List.of(mockInputFile(pyFile)));

    assertThat(result.roots()).containsExactly(baseDir.getAbsolutePath());
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.LEGACY_INIT_PY);
  }

  @Test
  void resolve_noBuildConfigFiles_multiplePackages_multipleRootsSortedDeepestFirst() throws IOException {
    // Two independent packages at different depths → two roots, sorted deepest first
    File baseDir = tempDir.toFile();

    File shallowDir = tempDir.resolve("shallow").toFile();
    shallowDir.mkdir();
    new File(shallowDir, "__init__.py").createNewFile();
    File shallowFile = new File(shallowDir, "mod.py");
    shallowFile.createNewFile();

    File deepParentDir = tempDir.resolve("deep").toFile();
    deepParentDir.mkdir();
    File deepPkgDir = new File(deepParentDir, "nested");
    deepPkgDir.mkdir();
    new File(deepPkgDir, "__init__.py").createNewFile();
    File deepFile = new File(deepPkgDir, "mod.py");
    deepFile.createNewFile();

    PackageResolutionResult result = PackageRootResolver.resolve(
      mockFileSystem(baseDir), noSonarSources(),
      List.of(mockInputFile(shallowFile), mockInputFile(deepFile)));

    // shallow/ has __init__.py → root = baseDir
    // deep/nested/ has __init__.py, deep/ does not → root = deep/
    // Roots sorted deepest-first: deep/ (deeper path) before baseDir
    assertThat(result.roots()).containsExactly(
      deepParentDir.getAbsolutePath(),
      baseDir.getAbsolutePath());
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.LEGACY_INIT_PY);
  }

  @Test
  void resolve_noBuildConfigFiles_nestedInitPyChain_walksUpToCorrectRoot() throws IOException {
    // level1/__init__.py, level1/level2/__init__.py, level1/level2/level3/__init__.py
    // file at level3/ → root should be baseDir
    File baseDir = tempDir.toFile();
    File l1 = new File(baseDir, "level1");
    File l2 = new File(l1, "level2");
    File l3 = new File(l2, "level3");
    l3.mkdirs();
    new File(l1, "__init__.py").createNewFile();
    new File(l2, "__init__.py").createNewFile();
    new File(l3, "__init__.py").createNewFile();
    File pyFile = new File(l3, "mod.py");
    pyFile.createNewFile();

    PackageResolutionResult result = PackageRootResolver.resolve(
      mockFileSystem(baseDir), noSonarSources(), List.of(mockInputFile(pyFile)));

    assertThat(result.roots()).containsExactly(baseDir.getAbsolutePath());
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.LEGACY_INIT_PY);
  }

  @Test
  void resolve_noBuildConfigFiles_initPyGap_stopsAtGap() throws IOException {
    // top/ (no __init__.py), top/mid/__init__.py, top/mid/bottom/__init__.py
    // file at bottom/ → root = top/ (first parent without __init__.py)
    File baseDir = tempDir.toFile();
    File top = new File(baseDir, "top");
    File mid = new File(top, "mid");
    File bottom = new File(mid, "bottom");
    bottom.mkdirs();
    new File(mid, "__init__.py").createNewFile();
    new File(bottom, "__init__.py").createNewFile();
    File pyFile = new File(bottom, "mod.py");
    pyFile.createNewFile();

    PackageResolutionResult result = PackageRootResolver.resolve(
      mockFileSystem(baseDir), noSonarSources(), List.of(mockInputFile(pyFile)));

    // bottom has __init__.py → up to mid (has __init__.py) → up to top (no __init__.py) → root = top
    assertThat(result.roots()).containsExactly(top.getAbsolutePath());
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.LEGACY_INIT_PY);
  }

  @Test
  void resolve_noBuildConfigFiles_fileAtBaseDir_producesBaseDirAsRoot() throws IOException {
    File baseDir = tempDir.toFile();
    File pyFile = new File(baseDir, "script.py");
    pyFile.createNewFile();

    PackageResolutionResult result = PackageRootResolver.resolve(
      mockFileSystem(baseDir), noSonarSources(), List.of(mockInputFile(pyFile)));

    // File is directly in baseDir, which has no __init__.py → root = baseDir
    assertThat(result.roots()).containsExactly(baseDir.getAbsolutePath());
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.LEGACY_INIT_PY);
  }

  // ─── resolve() — build config files present ───────────────────────────────

  @Test
  void resolve_buildFileExistsNoRoots_conventionalFoldersTakePriorityOverSonarSources() throws IOException {
    // pyproject.toml exists but provides no source roots; src/ exists; sonar.sources set
    // => conventional folders should win (build file present path)
    Files.createDirectory(tempDir.resolve("src"));
    Files.writeString(tempDir.resolve("pyproject.toml"), "[project]\nname = \"myproject\"\n");
    File baseDir = tempDir.toFile();

    PackageResolutionResult result = PackageRootResolver.resolve(mockFileSystem(baseDir), sonarSources("app"), List.of());

    assertThat(result.roots()).containsExactly(new File(baseDir, "src").getAbsolutePath());
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.CONVENTIONAL_FOLDERS);
  }

  @Test
  void resolve_buildFileExistsNoRoots_fallsBackToSonarSources() throws IOException {
    // pyproject.toml exists but provides no source roots; no conventional folders; sonar.sources set
    Files.writeString(tempDir.resolve("pyproject.toml"), "[project]\nname = \"myproject\"\n");
    File baseDir = tempDir.toFile();

    PackageResolutionResult result = PackageRootResolver.resolve(mockFileSystem(baseDir), sonarSources("mysrc"), List.of());

    assertThat(result.roots()).containsExactly(new File(baseDir, "mysrc").getAbsolutePath());
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.SONAR_SOURCES);
  }

  @Test
  void resolve_setupPyExistsNoRoots_conventionalFoldersTakePriorityOverSonarSources() throws IOException {
    // setup.py exists but provides no source roots; src/ exists; sonar.sources set
    Files.createDirectory(tempDir.resolve("src"));
    Files.writeString(tempDir.resolve("setup.py"), "# empty\n");
    File baseDir = tempDir.toFile();

    PackageResolutionResult result = PackageRootResolver.resolve(mockFileSystem(baseDir), sonarSources("app"), List.of());

    assertThat(result.roots()).containsExactly(new File(baseDir, "src").getAbsolutePath());
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.CONVENTIONAL_FOLDERS);
  }

  @Test
  void resolve_conventionalFoldersFallback_augmentsWithUncoveredFileRoots() throws IOException {
    // pyproject.toml exists but provides no source roots; src/ exists (conventional folder);
    // a test file lives under tests/ with __init__.py (outside src/).
    // The fallback should augment the conventional root with a legacy root for the test file.
    File baseDir = tempDir.toFile();
    Files.createDirectory(tempDir.resolve("src"));
    Files.writeString(tempDir.resolve("pyproject.toml"), "[project]\nname = \"myproject\"\n");

    File testsDir = tempDir.resolve("tests").toFile();
    testsDir.mkdir();
    new File(testsDir, "__init__.py").createNewFile();
    File testFile = new File(testsDir, "test_app.py");
    testFile.createNewFile();

    PackageResolutionResult result = PackageRootResolver.resolve(
      mockFileSystem(baseDir), noSonarSources(), List.of(mockInputFile(testFile)));

    // src/ from conventional folders + baseDir from legacy walk-up (tests/ has __init__.py)
    assertThat(result.roots()).containsExactlyInAnyOrder(
      new File(baseDir, "src").getAbsolutePath(),
      baseDir.getAbsolutePath());
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.CONVENTIONAL_FOLDERS);

    // Verify FQN for the test file
    String testPackageName = org.sonar.python.semantic.SymbolUtils.pythonPackageName(testFile, result.roots());
    assertThat(testPackageName).isEqualTo("tests");
  }

  @Test
  void resolve_sonarSourcesFallback_augmentsWithUncoveredFileRoots() throws IOException {
    // pyproject.toml exists but provides no source roots; no conventional folders;
    // sonar.sources=app; a test file lives under tests/ with __init__.py (outside app/).
    File baseDir = tempDir.toFile();
    Files.writeString(tempDir.resolve("pyproject.toml"), "[project]\nname = \"myproject\"\n");

    File testsDir = tempDir.resolve("tests").toFile();
    testsDir.mkdir();
    new File(testsDir, "__init__.py").createNewFile();
    File testFile = new File(testsDir, "test_app.py");
    testFile.createNewFile();

    PackageResolutionResult result = PackageRootResolver.resolve(
      mockFileSystem(baseDir), sonarSources("app"), List.of(mockInputFile(testFile)));

    // app/ from sonar.sources + baseDir from legacy walk-up (tests/ has __init__.py)
    assertThat(result.roots()).containsExactlyInAnyOrder(
      new File(baseDir, "app").getAbsolutePath(),
      baseDir.getAbsolutePath());
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.SONAR_SOURCES);

    // Verify FQN for the test file
    String testPackageName = org.sonar.python.semantic.SymbolUtils.pythonPackageName(testFile, result.roots());
    assertThat(testPackageName).isEqualTo("tests");
  }

  // ─── toAbsolutePaths() — normalization ────────────────────────────────────

  @Test
  void toAbsolutePaths_normalPaths_resolvedCorrectly() {
    File baseDir = tempDir.toFile();
    List<String> result = PackageRootResolver.toAbsolutePaths(List.of("app", "core"), baseDir);

    assertThat(result).containsExactly(
      new File(baseDir, "app").getAbsolutePath(),
      new File(baseDir, "core").getAbsolutePath());
  }

  @Test
  void toAbsolutePaths_dotPath_normalizedToBaseDir() {
    File baseDir = tempDir.toFile();
    List<String> result = PackageRootResolver.toAbsolutePaths(List.of("."), baseDir);

    // "." must normalize to baseDir, not "/baseDir/."
    assertThat(result).containsExactly(baseDir.getAbsolutePath());
    assertThat(result.get(0)).doesNotEndWith("/.");
    assertThat(result.get(0)).doesNotEndWith("\\.");
  }

  // ─── toAbsolutePaths() — Windows path handling ───────────────────────────

  @Test
  void toAbsolutePaths_windowsAbsolutePath_notPrependedWithBaseDir() {
    // When sonar.sources contains a Windows-style absolute path (e.g. from a Windows scanner),
    // it must NOT be prepended with the base directory on a non-Windows system.
    File baseDir = new File("/project/base");
    String windowsAbsPath = "C:\\Users\\user\\mixed-language-build";

    List<String> result = PackageRootResolver.toAbsolutePaths(List.of(windowsAbsPath), baseDir);

    // Must not produce "/project/base/" prepended to the Windows path
    assertThat(result).containsExactly("C:\\Users\\user\\mixed-language-build");
  }

  @Test
  void toAbsolutePaths_windowsAbsolutePathWithForwardSlashes_notPrependedWithBaseDir() {
    // Windows paths may also use forward slashes; they are normalised by Path.normalize()
    File baseDir = new File("/project/base");
    String windowsAbsPath = "C:/Users/user/mixed-language-build";

    List<String> result = PackageRootResolver.toAbsolutePaths(List.of(windowsAbsPath), baseDir);

    // Must not produce "/project/base/" prepended to the Windows path
    // Forward slashes are preserved as-is by Path.normalize() on non-Windows systems
    assertThat(result).hasSize(1);
    assertThat(result.get(0)).isIn("C:\\Users\\user\\mixed-language-build", "C:/Users/user/mixed-language-build");
  }

  @Test
  void trySonarSources_windowsAbsolutePath_usedDirectlyNotPrependedWithBaseDir() throws IOException {
    // Full integration: build file exists, sonar.sources has a Windows absolute path.
    // The resolved root should be derived from the Windows path, not baseDir + Windows path.
    Files.writeString(tempDir.resolve("pyproject.toml"), "[project]\nname = \"myproject\"\n");
    File baseDir = tempDir.toFile();
    String windowsAbsPath = "C:\\Users\\user\\mixed-language-build";

    PackageResolutionResult result = PackageRootResolver.resolve(mockFileSystem(baseDir), sonarSources(windowsAbsPath), List.of());

    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.SONAR_SOURCES);
    assertThat(result.roots()).hasSize(1);
    assertThat(result.roots()).containsExactly("C:\\Users\\user\\mixed-language-build");
  }

  // ─── adjustPackageRoot() — migrated from PythonIndexerTest ────────────────

  @Test
  void adjustPackageRoot_noInitPy(@TempDir Path localTempDir) {
    File baseDir = localTempDir.toFile();
    File srcDir = new File(baseDir, "src");
    srcDir.mkdir();

    String result = PackageRootResolver.adjustPackageRoot(srcDir, baseDir);
    assertThat(result).isEqualTo(srcDir.getAbsolutePath());
  }

  @Test
  void adjustPackageRoot_preservesBaseDirAliasForDerivedRoot(@TempDir Path localTempDir) throws IOException {
    Path realBaseDir = Files.createDirectories(localTempDir.resolve("real"));
    Path derivedRoot = Files.createDirectories(realBaseDir.resolve("tests/unit"));
    Path aliasBaseDir = localTempDir.resolve("alias");
    try {
      Files.createSymbolicLink(aliasBaseDir, realBaseDir);
    } catch (UnsupportedOperationException | IOException | SecurityException e) {
      Assumptions.assumeTrue(false, "Symbolic links are not available: " + e.getMessage());
    }

    String result = PackageRootResolver.adjustPackageRoot(derivedRoot.toFile(), aliasBaseDir.toFile());

    assertThat(result).isEqualTo(aliasBaseDir.resolve("tests/unit").toAbsolutePath().toString());
  }

  @Test
  void adjustPackageRoot_preservesRootOutsideBaseDir(@TempDir Path localTempDir) throws IOException {
    File baseDir = Files.createDirectory(localTempDir.resolve("project")).toFile();
    File externalRoot = Files.createDirectory(localTempDir.resolve("external")).toFile();

    String result = PackageRootResolver.adjustPackageRoot(externalRoot, baseDir);

    assertThat(result).isEqualTo(externalRoot.getAbsolutePath());
  }

  @Test
  void adjustPackageRoot_withInitPy_walksUp(@TempDir Path localTempDir) throws Exception {
    File baseDir = localTempDir.toFile();
    File srcDir = new File(baseDir, "src");
    File packageDir = new File(srcDir, "mypackage");
    packageDir.mkdirs();
    new File(packageDir, "__init__.py").createNewFile();

    String result = PackageRootResolver.adjustPackageRoot(packageDir, baseDir);
    assertThat(result).isEqualTo(srcDir.getAbsolutePath());
  }

  @Test
  void adjustPackageRoot_nestedPackagesWithInitPy(@TempDir Path localTempDir) throws Exception {
    File baseDir = localTempDir.toFile();
    File srcDir = new File(baseDir, "src");
    File level1 = new File(srcDir, "level1");
    File level2 = new File(level1, "level2");
    File level3 = new File(level2, "level3");
    level3.mkdirs();

    new File(level1, "__init__.py").createNewFile();
    new File(level2, "__init__.py").createNewFile();
    new File(level3, "__init__.py").createNewFile();

    String result = PackageRootResolver.adjustPackageRoot(level3, baseDir);
    assertThat(result).isEqualTo(srcDir.getAbsolutePath());
  }

  @Test
  void adjustPackageRoot_stopsAtBaseDir(@TempDir Path localTempDir) throws Exception {
    File baseDir = localTempDir.toFile();
    File level1 = new File(baseDir, "level1");
    File level2 = new File(level1, "level2");
    level2.mkdirs();
    new File(baseDir, "__init__.py").createNewFile();
    new File(level1, "__init__.py").createNewFile();
    new File(level2, "__init__.py").createNewFile();

    String result = PackageRootResolver.adjustPackageRoot(level2, baseDir);
    assertThat(result).isEqualTo(baseDir.getAbsolutePath());
  }

  @Test
  void adjustPackageRoot_rootEqualsBaseDir(@TempDir Path localTempDir) throws Exception {
    File baseDir = localTempDir.toFile();
    new File(baseDir, "__init__.py").createNewFile();

    String result = PackageRootResolver.adjustPackageRoot(baseDir, baseDir);
    assertThat(result).isEqualTo(baseDir.getAbsolutePath());
  }

  @Test
  void adjustPackageRoot_partialInitPyChain(@TempDir Path localTempDir) throws Exception {
    File baseDir = localTempDir.toFile();
    File srcDir = new File(baseDir, "src");
    File withInit = new File(srcDir, "withInit");
    File withoutInit = new File(withInit, "withoutInit");
    File deepPackage = new File(withoutInit, "deepPackage");
    deepPackage.mkdirs();

    new File(withInit, "__init__.py").createNewFile();
    new File(deepPackage, "__init__.py").createNewFile();

    String result = PackageRootResolver.adjustPackageRoot(deepPackage, baseDir);
    assertThat(result).isEqualTo(withoutInit.getAbsolutePath());
  }

  @Test
  void adjustPackageRoot_emptyDirectory(@TempDir Path localTempDir) {
    File baseDir = localTempDir.toFile();
    File emptyDir = new File(baseDir, "empty");
    emptyDir.mkdir();

    String result = PackageRootResolver.adjustPackageRoot(emptyDir, baseDir);
    assertThat(result).isEqualTo(emptyDir.getAbsolutePath());
  }

  @Test
  void adjustPackageRoot_singleLevelWithInitPy(@TempDir Path localTempDir) throws Exception {
    File baseDir = localTempDir.toFile();
    File packageDir = new File(baseDir, "mypackage");
    packageDir.mkdir();
    new File(packageDir, "__init__.py").createNewFile();

    String result = PackageRootResolver.adjustPackageRoot(packageDir, baseDir);
    assertThat(result).isEqualTo(baseDir.getAbsolutePath());
  }

  // ─── computePackageRootsFromFiles() ──────────────────────────────────────

  @Test
  void computePackageRootsFromFiles_empty_returns_baseDir(@TempDir Path localTempDir) {
    File baseDir = localTempDir.toFile();
    List<String> roots = PackageRootResolver.computePackageRootsFromFiles(List.of(), baseDir);
    assertThat(roots).containsExactly(baseDir.getAbsolutePath());
  }

  @Test
  void computePackageRootsFromFiles_walks_up_init_py_chain(@TempDir Path localTempDir) throws Exception {
    // pkg/__init__.py exists → pkg/ is a package, root should be localTempDir (parent of pkg)
    File pkgDir = localTempDir.resolve("pkg").toFile();
    pkgDir.mkdirs();
    new File(pkgDir, "__init__.py").createNewFile();
    File module = new File(pkgDir, "mod.py");
    module.createNewFile();

    List<String> roots = PackageRootResolver.computePackageRootsFromFiles(List.of(module), localTempDir.toFile());
    assertThat(roots).containsExactly(localTempDir.toFile().getAbsolutePath());
  }

  // ─── augmentWithUncoveredFileRoots — files outside build-config roots ───

  @Test
  void resolve_setupPyWithPackageDir_testsDirOutsideSrcRoot_augmentedWithLegacyRoot() throws IOException {
    // Reproduces the 'black' project scenario:
    // - setup.py declares package_dir={"": "src"}
    // - Tests live under tests/ with __init__.py (outside src/)
    // - The resolver must add a root for tests/ files so they get correct package names
    File baseDir = tempDir.toFile();

    File srcDir = tempDir.resolve("src").toFile();
    File blackPkg = new File(srcDir, "black");
    blackPkg.mkdirs();
    new File(blackPkg, "__init__.py").createNewFile();
    File srcModule = new File(blackPkg, "linegen.py");
    srcModule.createNewFile();

    File testsDir = tempDir.resolve("tests").toFile();
    testsDir.mkdirs();
    new File(testsDir, "__init__.py").createNewFile();
    File testFile = new File(testsDir, "test_black.py");
    testFile.createNewFile();

    Files.writeString(tempDir.resolve("setup.py"), """
        from setuptools import setup, find_packages
        setup(packages=find_packages(where='src'), package_dir={'': 'src'})""");

    List<InputFile> pythonFiles = List.of(mockInputFile(srcModule), mockInputFile(testFile));
    PackageResolutionResult result = PackageRootResolver.resolve(mockFileSystem(baseDir), noSonarSources(), pythonFiles);

    assertThat(result.roots()).contains(srcDir.getAbsolutePath());
    // tests/ has __init__.py so the legacy root for test_black.py is baseDir
    assertThat(result.roots()).contains(baseDir.getAbsolutePath());

    // Verify that test_black.py gets the correct package name
    String testPackageName = org.sonar.python.semantic.SymbolUtils.pythonPackageName(testFile, result.roots());
    assertThat(testPackageName).isEqualTo("tests");
  }

  @Test
  void resolve_setupPyWithPackageDir_allFilesCovered_noAugmentation() throws IOException {
    // When all files are under the build-config root, no extra roots are added
    File baseDir = tempDir.toFile();

    File srcDir = tempDir.resolve("src").toFile();
    File pkg = new File(srcDir, "mypackage");
    pkg.mkdirs();
    new File(pkg, "__init__.py").createNewFile();
    File module = new File(pkg, "mod.py");
    module.createNewFile();

    Files.writeString(tempDir.resolve("setup.py"),"""
        from setuptools import setup, find_packages
        setup(packages=find_packages(where='src'), package_dir={'': 'src'})""");

    List<InputFile> pythonFiles = List.of(mockInputFile(module));
    PackageResolutionResult result = PackageRootResolver.resolve(mockFileSystem(baseDir), noSonarSources(), pythonFiles);

    assertThat(result.roots()).containsExactly(srcDir.getAbsolutePath());
  }

  @Test
  void resolve_setupPyWithMultipleRoots_allFilesCovered_rootsSortedDeepestFirst() throws IOException {
    // When all files are covered (no augmentation), roots must still be sorted deepest-first.
    // This verifies the fix for the early-return path in augmentWithUncoveredFileRoots.
    File baseDir = tempDir.toFile();

    // Create two source roots at different depths: src/ (shallow) and src/extra/nested/ (deep)
    File srcDir = tempDir.resolve("src").toFile();
    File shallowPkg = new File(srcDir, "shallowpkg");
    shallowPkg.mkdirs();
    new File(shallowPkg, "__init__.py").createNewFile();
    File shallowModule = new File(shallowPkg, "mod.py");
    shallowModule.createNewFile();

    File extraNested = new File(srcDir, "extra/nested");
    File deepPkg = new File(extraNested, "deeppkg");
    deepPkg.mkdirs();
    new File(deepPkg, "__init__.py").createNewFile();
    File deepModule = new File(deepPkg, "mod.py");
    deepModule.createNewFile();

    // setup.py declares both src/ and src/extra/nested/ as package dirs
    Files.writeString(tempDir.resolve("setup.py"), """
        from setuptools import setup, find_packages
        setup(
            packages=find_packages(where='src') + find_packages(where='src/extra/nested'),
            package_dir={'': 'src', 'deeppkg': 'src/extra/nested/deeppkg'},
        )""");

    // All files are under declared roots, so no augmentation occurs
    List<InputFile> pythonFiles = List.of(mockInputFile(shallowModule), mockInputFile(deepModule));
    PackageResolutionResult result = PackageRootResolver.resolve(mockFileSystem(baseDir), noSonarSources(), pythonFiles);

    // Roots must be sorted deepest-first: src/extra/nested before src
    assertThat(result.roots()).containsExactly(
      extraNested.getAbsolutePath(),
      srcDir.getAbsolutePath());
  }

  @Test
  void computePackageRootsFromFiles_multiple_roots_sorted_deepest_first(@TempDir Path localTempDir) throws Exception {
    // pkgA/__init__.py → root = localTempDir (shallow)
    File pkgA = localTempDir.resolve("pkgA").toFile();
    pkgA.mkdirs();
    new File(pkgA, "__init__.py").createNewFile();
    File modA = new File(pkgA, "mod.py");
    modA.createNewFile();

    // dir/pkgB/__init__.py → root = dir/ (deeper path)
    File dir = localTempDir.resolve("dir").toFile();
    File pkgB = new File(dir, "pkgB");
    pkgB.mkdirs();
    new File(pkgB, "__init__.py").createNewFile();
    File modB = new File(pkgB, "mod.py");
    modB.createNewFile();

    List<String> roots = PackageRootResolver.computePackageRootsFromFiles(List.of(modA, modB), localTempDir.toFile());
    // dir/ is deeper (more separators) than localTempDir → dir/ comes first
    assertThat(roots).containsExactly(
      dir.getAbsolutePath(),
      localTempDir.toFile().getAbsolutePath());
  }

  @Test
  void computePackageRootsFromFiles_same_depth_sorted_alphabetically(@TempDir Path localTempDir) throws Exception {
    // Two roots at the same depth should be sorted alphabetically
    File baseDir = localTempDir.toFile();

    // beta/pkgB/__init__.py → root = beta/
    File betaDir = localTempDir.resolve("beta").toFile();
    File pkgB = new File(betaDir, "pkgB");
    pkgB.mkdirs();
    new File(pkgB, "__init__.py").createNewFile();
    File modB = new File(pkgB, "mod.py");
    modB.createNewFile();

    // alpha/pkgA/__init__.py → root = alpha/
    File alphaDir = localTempDir.resolve("alpha").toFile();
    File pkgA = new File(alphaDir, "pkgA");
    pkgA.mkdirs();
    new File(pkgA, "__init__.py").createNewFile();
    File modA = new File(pkgA, "mod.py");
    modA.createNewFile();

    // Pass files in reverse alphabetical order to verify sorting is applied
    List<String> roots = PackageRootResolver.computePackageRootsFromFiles(List.of(modB, modA), baseDir);
    // Same depth → alphabetical: alpha/ before beta/
    assertThat(roots).containsExactly(
      alphaDir.getAbsolutePath(),
      betaDir.getAbsolutePath());
  }
}
