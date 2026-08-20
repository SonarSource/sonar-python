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

import static org.sonar.python.semantic.SymbolUtils.resolvedPath;

import com.google.common.annotations.VisibleForTesting;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.fs.FileSystem;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.config.Configuration;

/**
 * Resolves package root directories for Python projects.
 *
 * <p>This class is the single source of truth for package root resolution. It handles:
 * <ul>
 *   <li>Extraction from pyproject.toml build system configurations</li>
 *   <li>Extraction from setup.py configurations</li>
 *   <li>When no build config files exist: computes roots by scanning Python source files
 *       and walking up __init__.py chains to find the first parent directory without __init__.py
 *       for each file (legacy detection mode)</li>
 *   <li>Fallback when build files exist but provide no roots: conventional folders (src/, lib/),
 *       then sonar.sources, then base directory</li>
 * </ul>
 */
public class PackageRootResolver {

  private static final Logger LOG = LoggerFactory.getLogger(PackageRootResolver.class);

  static final String SONAR_SOURCES_KEY = "sonar.sources";
  static final List<String> CONVENTIONAL_FOLDERS = List.of("src", "lib");

  private PackageRootResolver() {
  }

  /**
   * Resolves package root directories for the project.
   *
   * <p>Attempts to extract source roots from pyproject.toml and setup.py build system configurations.
   * When no build config files exist, computes roots by scanning the provided Python source files
   * and walking up __init__.py chains to find the package root for each file. When build config
   * files exist but provide no source roots, conventional folders take priority over sonar.sources,
   * then base directory as last resort.
   *
   * @param fileSystem the Sonar file system providing the base directory
   * @param config the Sonar configuration
   * @param pythonFiles the Python source files in the project (used for legacy root computation)
   * @return resolution result including resolved root absolute paths and method information
   */
  public static PackageResolutionResult resolve(FileSystem fileSystem, Configuration config, Iterable<InputFile> pythonFiles) {
    File baseDir = fileSystem.baseDir();

    // Discover build config files
    BuildConfigFiles buildConfigFiles = findBuildConfigFiles(fileSystem);
    List<File> pyprojectFiles = buildConfigFiles.pyprojectFiles();
    List<File> setupPyFiles = buildConfigFiles.setupPyFiles();
    boolean hasBuildConfigFiles = !pyprojectFiles.isEmpty() || !setupPyFiles.isEmpty();

    // When no build config files exist, compute roots from __init__.py chains in source files.
    if (!hasBuildConfigFiles) {
      List<String> legacyRoots = computeLegacyRoots(pythonFiles, baseDir);
      LOG.debug("No build config files found; computed package roots from __init__.py detection: {}", legacyRoots);
      return PackageResolutionResult.fromLegacyInitPy(legacyRoots);
    }

    // Extract source roots from discovered files
    List<PyProjectExtractionResult> pyprojectResults = pyprojectFiles.stream()
      .map(PyProjectTomlSourceRoots::extractWithBuildSystem)
      .filter(PyProjectExtractionResult::hasRoots)
      .toList();
    List<ConfigSourceRoots> setupPyRoots = setupPyFiles.stream()
      .map(SetupPySourceRoots::extractWithLocation)
      .filter(csr -> !csr.relativeRoots().isEmpty())
      .toList();

    boolean hasPyprojectRoots = pyprojectResults.stream().anyMatch(PyProjectExtractionResult::hasRoots);
    boolean hasSetupPyRoots = !setupPyRoots.isEmpty();

    List<String> combinedRoots = Stream.concat(
        pyprojectResults.stream().map(PyProjectExtractionResult::configRoots).flatMap(crs -> crs.toAbsolutePaths().stream()),
        setupPyRoots.stream().flatMap(csr -> csr.toAbsolutePaths().stream()))
      .distinct()
      .toList();

    // Build configs exist but extracted no roots — fall back, then augment with uncovered files.
    if (!hasPyprojectRoots && !hasSetupPyRoots) {
      return resolveRootsWhenBuildConfigHasNoSourceRoots(config, baseDir, pythonFiles);
    }

    List<String> adjustedRoots = augmentWithUncoveredFileRoots(adjustRoots(combinedRoots, baseDir), pythonFiles, baseDir);
    LOG.debug("Resolved package roots from build configuration: {}", adjustedRoots);

    if (hasPyprojectRoots && hasSetupPyRoots) {
      return PackageResolutionResult.fromBothPyProjectAndSetupPy(adjustedRoots, getCombinedBuildSystem(pyprojectResults));
    }

    if (hasPyprojectRoots) {
      return PackageResolutionResult.fromPyProjectToml(adjustedRoots, getCombinedBuildSystem(pyprojectResults));
    }

    return PackageResolutionResult.fromSetupPy(adjustedRoots);
  }

  /**
   * Resolves fallback package roots when build config files exist but provide no source roots.
   *
   * <p>Priority order: conventional folders (src/, lib/), then sonar.sources, then base directory.
   * After resolving the primary roots, augments with legacy __init__.py-based roots for any
   * Python files not covered by the resolved roots.
   */
  private static PackageResolutionResult resolveRootsWhenBuildConfigHasNoSourceRoots(Configuration config, File baseDir, Iterable<InputFile> pythonFiles) {
    List<BiFunction<Configuration, File, Optional<PackageResolutionResult>>> candidates =
      List.of(PackageRootResolver::tryConventionalFolders, PackageRootResolver::trySonarSources);
    for (BiFunction<Configuration, File, Optional<PackageResolutionResult>> candidate : candidates) {
      Optional<PackageResolutionResult> result = candidate.apply(config, baseDir);
      if (result.isPresent()) {
        var augmentedRoots = augmentWithUncoveredFileRoots(result.get().roots(), pythonFiles, baseDir);
        return new PackageResolutionResult(augmentedRoots, result.get().method(), result.get().buildSystem());
      }
    }

    LOG.debug("Using project base directory as package root (fallback)");
    return PackageResolutionResult.fromBaseDir(List.of(baseDir.getAbsolutePath()));
  }

  private static Optional<PackageResolutionResult> tryConventionalFolders(Configuration config, File baseDir) {
    List<String> conventionalFolders = findConventionalFolders(baseDir);
    if (conventionalFolders.isEmpty()) {
      return Optional.empty();
    }
    List<String> adjustedRoots = adjustRoots(toAbsolutePaths(conventionalFolders, baseDir), baseDir);
    LOG.debug("Resolved package roots from fallback (conventional folders): {}", adjustedRoots);
    return Optional.of(PackageResolutionResult.fromConventionalFolders(adjustedRoots));
  }

  private static Optional<PackageResolutionResult> trySonarSources(Configuration config, File baseDir) {
    String[] sonarSources = config.getStringArray(SONAR_SOURCES_KEY);
    if (sonarSources.length == 0) {
      return Optional.empty();
    }
    List<String> adjustedRoots = adjustRoots(toAbsolutePaths(Arrays.asList(sonarSources), baseDir), baseDir);
    LOG.debug("Resolved package roots from fallback (sonar.sources): {}", adjustedRoots);
    return Optional.of(PackageResolutionResult.fromSonarSources(adjustedRoots));
  }

  /**
   * Converts path strings to normalized absolute paths.
   *
   * <p>Relative paths are resolved against the given base directory. Windows-style absolute paths
   * (e.g. {@code C:\path\to\src} or {@code C:/path/to/src}) are used as-is without prepending the
   * base directory, which would otherwise produce a nonsensical path on non-Windows systems.
   *
   * <p>Uses {@link Path#normalize()} to resolve {@code .} and {@code ..} components without
   * performing any I/O, so that {@code sonar.sources=.} correctly resolves to the base directory
   * rather than producing an un-normalized path like {@code /project/.}.
   */
  static List<String> toAbsolutePaths(List<String> paths, File baseDir) {
    return paths.stream()
      .map(path -> {
        File file = isWindowsAbsolutePath(path) ? new File(path) : new File(baseDir, path);
        return file.toPath().normalize().toString();
      })
      .toList();
  }

  /**
   * Returns {@code true} if the given path is a Windows-style absolute path (e.g. {@code C:\...}
   * or {@code C:/...}), regardless of the current operating system.
   */
  private static boolean isWindowsAbsolutePath(String path) {
    return path.length() >= 3
      && Character.isLetter(path.charAt(0))
      && path.charAt(1) == ':'
      && (path.charAt(2) == '\\' || path.charAt(2) == '/');
  }

  private static List<String> findConventionalFolders(File baseDir) {
    List<String> folders = new ArrayList<>();
    for (String folderName : CONVENTIONAL_FOLDERS) {
      File folder = new File(baseDir, folderName);
      if (folder.exists() && folder.isDirectory()) {
        folders.add(folderName);
      }
    }
    return folders;
  }

  private static List<String> adjustRoots(List<String> roots, File baseDir) {
    return roots.stream()
      .map(root -> {
        File rootAsFile = new File(root);
        if (rootAsFile.isAbsolute()) {
          // Native absolute path (works on any OS, including Windows running on Windows).
          return adjustPackageRoot(rootAsFile, baseDir);
        }
        if (isWindowsAbsolutePath(root)) {
          // Windows-style absolute path (e.g. C:\src) on a non-Windows system: File.isAbsolute()
          // returns false, so we must not pass it to new File(baseDir, root) or getAbsolutePath()
          // would prepend the JVM working directory. Return as-is; __init__.py traversal is not
          // meaningful for a foreign-OS path.
          return root;
        }
        return adjustPackageRoot(new File(baseDir, root), baseDir);
      })
      .distinct()
      .toList();
  }

  /**
   * Adjusts a package root by walking up the directory tree if it contains __init__.py.
   *
   * <p>If the root directory contains __init__.py, it's part of a package, not the package root.
   * We walk up to find the first parent directory without __init__.py.
   *
   * @param root    the potential package root directory
   * @param baseDir the project base directory (we don't walk above this)
   * @return the adjusted package root absolute path
   */
  @VisibleForTesting
  static String adjustPackageRoot(File root, File baseDir) {
    File current = root;
    String baseDirPath = baseDir.getAbsolutePath();
    Path comparableBaseDirPath = resolvedPath(baseDir);
    while (current != null && !resolvedPath(current).equals(comparableBaseDirPath)) {
      File initFile = new File(current, "__init__.py");
      if (!initFile.exists()) {
        break;
      }
      current = current.getParentFile();
    }
    if (current == null || resolvedPath(current).equals(comparableBaseDirPath)) {
      return baseDirPath;
    }
    Path comparableCurrentPath = resolvedPath(current);
    if (comparableCurrentPath.startsWith(comparableBaseDirPath)) {
      Path relativePath = comparableBaseDirPath.relativize(comparableCurrentPath);
      return baseDir.toPath().toAbsolutePath().resolve(relativePath).normalize().toString();
    }
    return current.getAbsolutePath();
  }

  /**
   * Computes package root directories for a set of files by walking up __init__.py chains.
   *
   * <p>For each file, walks up from the file's parent directory toward the base directory,
   * stopping at the first directory that does not contain an __init__.py. That directory is the
   * package root for that file. Roots are deduplicated and sorted by path length descending so
   * that the most specific root is matched first.
   *
   * <p>If no files are provided, returns the base directory as the single root.
   *
   * @param files   the Python source files whose roots to compute
   * @param baseDir the project base directory (walk-up never crosses this boundary)
   * @return deduplicated absolute package root paths, sorted deepest-first
   */
  public static List<String> computePackageRootsFromFiles(List<File> files, File baseDir) {
    List<String> roots = files.stream()
      .map(f -> adjustPackageRoot(f.getParentFile(), baseDir))
      .distinct()
      .sorted(deepestFirstComparator())
      .toList();
    return roots.isEmpty() ? List.of(baseDir.getAbsolutePath()) : roots;
  }

  /**
   * Augments build-config roots with legacy __init__.py-based roots for files not covered
   * by any existing root.
   *
   * <p>Build configuration files (setup.py, pyproject.toml) typically declare only the source
   * package roots (e.g. {@code src/}). Files outside those roots — most commonly test files
   * under a {@code tests/} directory that has its own {@code __init__.py} — would otherwise
   * receive an empty package name, breaking FQN resolution and type inference.
   *
   * <p>This method identifies such uncovered files and computes additional roots for them
   * using the same __init__.py walk-up algorithm used for legacy projects.
   */
  private static List<String> augmentWithUncoveredFileRoots(List<String> configRoots, Iterable<InputFile> pythonFiles, File baseDir) {
    List<Path> comparableRoots = configRoots.stream()
      .map(root -> resolvedPath(new File(root)))
      .toList();
    List<File> uncoveredFiles = StreamSupport.stream(pythonFiles.spliterator(), false)
      .map(InputFile::file)
      .filter(file -> !isUnderAnyComparableRoot(file, comparableRoots))
      .toList();

    if (uncoveredFiles.isEmpty()) {
      return configRoots.stream()
        .sorted(deepestFirstComparator())
        .toList();
    }

    List<String> additionalRoots = computePackageRootsFromFiles(uncoveredFiles, baseDir);
    return Stream.concat(configRoots.stream(), additionalRoots.stream())
      .distinct()
      .sorted(deepestFirstComparator())
      .toList();
  }

  private static boolean isUnderAnyComparableRoot(File file, List<Path> roots) {
    Path filePath = resolvedPath(file);
    for (Path root : roots) {
      if (filePath.startsWith(root)) {
        return true;
      }
    }
    return false;
  }

  private static List<String> computeLegacyRoots(Iterable<InputFile> pythonFiles, File baseDir) {
    List<File> files = StreamSupport.stream(pythonFiles.spliterator(), false)
      .map(InputFile::file)
      .toList();
    return computePackageRootsFromFiles(files, baseDir);
  }

  /**
   * Recursively finds supported build configuration files under the project base directory.
   */
  private static BuildConfigFiles findBuildConfigFiles(FileSystem fileSystem) {
    List<File> pyprojectFiles = new ArrayList<>();
    List<File> setupPyFiles = new ArrayList<>();
    try (Stream<Path> stream = Files.walk(fileSystem.baseDir().toPath())) {
      stream
        .filter(Files::isRegularFile)
        .forEach(path -> {
          String filename = path.getFileName().toString();
          if ("pyproject.toml".equals(filename)) {
            pyprojectFiles.add(path.toFile());
          } else if ("setup.py".equals(filename)) {
            setupPyFiles.add(path.toFile());
          }
        });
    } catch (IOException e) {
      return new BuildConfigFiles(List.of(), List.of());
    }
    return new BuildConfigFiles(List.copyOf(pyprojectFiles), List.copyOf(setupPyFiles));
  }

  private record BuildConfigFiles(List<File> pyprojectFiles, List<File> setupPyFiles) {
  }

  /**
   * Returns a comparator that sorts package root paths deepest-first (by separator count),
   * breaking ties alphabetically.
   */
  private static Comparator<String> deepestFirstComparator() {
    return Comparator.comparingInt(PackageRootResolver::pathDepth).reversed().thenComparing(Comparator.naturalOrder());
  }

  private static int pathDepth(String path) {
    return (int) path.chars().filter(c -> c == File.separatorChar).count();
  }

  /**
   * Determines the combined build system across multiple pyproject.toml results.
   * If multiple files report different build systems, returns MULTIPLE.
   */
  private static PackageResolutionResult.BuildSystem getCombinedBuildSystem(List<PyProjectExtractionResult> pyprojectResults) {
    return pyprojectResults.stream()
      .map(PyProjectExtractionResult::buildSystem)
      .filter(bs -> bs != PackageResolutionResult.BuildSystem.NONE)
      .distinct()
      .reduce((a, b) -> PackageResolutionResult.BuildSystem.MULTIPLE)
      .orElse(PackageResolutionResult.BuildSystem.NONE);
  }
}
