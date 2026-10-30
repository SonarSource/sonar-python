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

import com.sonarsource.scanner.engine.sensor.test.fixtures.SensorContextTester;
import java.io.IOException;
import java.nio.file.AccessDeniedException;
import java.nio.file.FileSystemException;
import java.nio.file.FileSystemLoopException;
import java.nio.file.FileVisitResult;
import java.nio.file.FileVisitor;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.slf4j.event.Level;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.testfixtures.log.LogTesterJUnit5;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.CALLS_REAL_METHODS;
import static org.mockito.Mockito.mockStatic;
import static org.sonar.plugins.python.TestUtils.createInputFile;

class PackageRootResolverPermissionsTest {

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5();

  @TempDir
  Path baseDir;

  @Test
  void readableDirectoryOutsideSourcesAndExcluded_resolvesNormally() throws IOException {
    SensorContextTester context = project();
    Files.createDirectories(baseDir.resolve("out/subdirectory"));

    assertSourceRoot(context);
  }

  @ParameterizedTest
  @CsvSource({"-wx-wx-wx, false", "---------, false", "-wx-wx-wx, true", "---------, true"})
  void unreadableDirectoryOutsideSourcesAndExcluded_doesNotAbortResolution(String permissions, boolean withBuildConfigs) throws IOException {
    SensorContextTester context = project();
    if (withBuildConfigs) {
      createBuildConfigs();
    }
    Path blocked = Files.createDirectories(baseDir.resolve("out/subdirectory"));
    assumeTrue(Files.getFileStore(blocked).supportsFileAttributeView("posix"));
    var originalPermissions = Files.getPosixFilePermissions(blocked);
    try {
      Files.setPosixFilePermissions(blocked, PosixFilePermissions.fromString(permissions));
      assumeFalse(Files.isReadable(blocked), "Run as a non-root user to reproduce denied directory access");

      if (withBuildConfigs) {
        assertBuildConfigRoots(context);
      } else {
        assertSourceRoot(context);
      }
      assertThat(logTester.logs(Level.WARN)).containsExactly(
        "Skipping inaccessible path '" + blocked + "' while searching for Python build configuration files.");
    } finally {
      Files.setPosixFilePermissions(blocked, originalPermissions);
    }
  }

  @ParameterizedTest
  @MethodSource("childIoFailures")
  void childIoFailure_retainsCollectedBuildConfigsAndContinuesDiscovery(IOException failure) throws IOException {
    assertContinuesAfterIoFailure(failure, false);
  }

  @ParameterizedTest
  @MethodSource("childIoFailures")
  void childDirectoryIterationFailure_retainsCollectedBuildConfigsAndContinuesDiscovery(IOException failure) throws IOException {
    assertContinuesAfterIoFailure(failure, true);
  }

  private static Stream<IOException> childIoFailures() {
    return Stream.of(
      new AccessDeniedException("child"),
      new NoSuchFileException("child"),
      new FileSystemException("child", null, "Operation not permitted"),
      new FileSystemLoopException("child"),
      new IOException("I/O failure"));
  }

  private void assertContinuesAfterIoFailure(IOException failure, boolean duringDirectoryIteration) throws IOException {
    SensorContextTester context = project();
    createBuildConfigs();
    Path firstConfig = baseDir.resolve("out/first/pyproject.toml");
    Path secondConfig = baseDir.resolve("out/second/setup.py");
    Path failedPath = baseDir.resolve("out/child");
    BasicFileAttributes firstAttributes = Files.readAttributes(firstConfig, BasicFileAttributes.class);
    BasicFileAttributes secondAttributes = Files.readAttributes(secondConfig, BasicFileAttributes.class);

    try (var files = mockStatic(Files.class, CALLS_REAL_METHODS)) {
      files.when(() -> Files.walkFileTree(eq(baseDir), any(FileVisitor.class))).thenAnswer(invocation -> {
        FileVisitor<Path> visitor = invocation.getArgument(1);
        visitor.visitFile(firstConfig, firstAttributes);
        FileVisitResult action = duringDirectoryIteration
          ? visitor.postVisitDirectory(failedPath, failure)
          : visitor.visitFileFailed(failedPath, failure);
        if (action == FileVisitResult.CONTINUE) {
          visitor.visitFile(secondConfig, secondAttributes);
        }
        return baseDir;
      });

      assertBuildConfigRoots(context);
    }
    assertThat(logTester.logs(Level.WARN)).containsExactly(
      "Skipping inaccessible path '" + failedPath + "' while searching for Python build configuration files.");
  }

  private void createBuildConfigs() throws IOException {
    Path pyproject = Files.createDirectories(baseDir.resolve("out/first"));
    Files.writeString(pyproject.resolve("pyproject.toml"), """
      [tool.setuptools.packages.find]
      where = ["pyproject-root"]
      """);
    Path setup = Files.createDirectories(baseDir.resolve("out/second"));
    Files.writeString(setup.resolve("setup.py"), """
      from setuptools import setup
      setup(package_dir={'': 'setup-root'})
      """);
  }

  private void assertBuildConfigRoots(SensorContextTester context) {
    PackageResolutionResult result = resolve(context);
    assertThat(result.roots()).containsExactlyInAnyOrder(
      baseDir.resolve("out/first/pyproject-root").toString(),
      baseDir.resolve("out/second/setup-root").toString(),
      baseDir.resolve("src").toString());
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.PYPROJECT_AND_SETUP_PY);
  }

  @ParameterizedTest
  @ValueSource(strings = {"-wx-wx-wx", "---------"})
  void unreadableBaseDirectory_retainsLegacyFallback(String permissions) throws IOException {
    SensorContextTester context = project();
    assumeTrue(Files.getFileStore(baseDir).supportsFileAttributeView("posix"));
    var originalPermissions = Files.getPosixFilePermissions(baseDir);
    try {
      Files.setPosixFilePermissions(baseDir, PosixFilePermissions.fromString(permissions));
      assumeFalse(Files.isReadable(baseDir), "Run as a non-root user to reproduce denied directory access");

      assertSourceRoot(context);
      assertThat(logTester.logs(Level.WARN)).isEmpty();
    } finally {
      Files.setPosixFilePermissions(baseDir, originalPermissions);
    }
  }

  private SensorContextTester project() throws IOException {
    Files.createDirectories(baseDir.resolve("src"));
    Files.writeString(baseDir.resolve("src/main.py"), "print('example')\n");
    SensorContextTester context = SensorContextTester.create(baseDir.toFile());
    context.settings().setProperty("sonar.sources", "src");
    context.settings().setProperty("sonar.exclusions", "out/**");
    context.fileSystem().add(createInputFile(baseDir.toFile(), "src/main.py", InputFile.Status.ADDED, InputFile.Type.MAIN).wrappedFile());
    return context;
  }

  private void assertSourceRoot(SensorContextTester context) {
    PackageResolutionResult result = resolve(context);
    assertThat(result.roots()).containsExactly(baseDir.resolve("src").toAbsolutePath().toString());
    assertThat(result.method()).isEqualTo(PackageResolutionResult.PrimaryResolutionMethod.LEGACY_INIT_PY);
  }

  private static PackageResolutionResult resolve(SensorContextTester context) {
    var fileSystem = context.fileSystem();
    var pythonFiles = fileSystem.inputFiles(fileSystem.predicates().hasLanguage("py"));
    return PackageRootResolver.resolve(fileSystem, context.config(), pythonFiles);
  }
}
