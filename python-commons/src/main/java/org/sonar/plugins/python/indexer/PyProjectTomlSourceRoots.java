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

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonSetter;
import com.fasterxml.jackson.annotation.Nulls;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.dataformat.toml.TomlMapper;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * Extracts source root directories from pyproject.toml build system configurations.
 *
 * <p>Supports the following build systems:
 * <ul>
 *   <li>setuptools: {@code [tool.setuptools.packages.find] where = ["src"]}</li>
 *   <li>Poetry: {@code [tool.poetry] packages = [{from = "src", include = "pkg"}]}</li>
 *   <li>Hatchling: {@code [tool.hatch.build.targets.wheel] sources = ["src"]}</li>
 *   <li>uv_build: {@code [build-system] build-backend = "uv_build"} or {@code [tool.uv.build-backend] module-root = "src"} - auto-detects src/ layout by convention</li>
 *   <li>PDM: {@code [tool.pdm] package-dir = "src"}</li>
 *   <li>Flit: auto-detects src/ layout by convention</li>
 * </ul>
 */
public class PyProjectTomlSourceRoots {

  private static final TomlMapper TOML_MAPPER = new TomlMapper();

  static {
    TOML_MAPPER.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
  }

  private PyProjectTomlSourceRoots() {
  }

  /**
   * Extracts source root directories from pyproject.toml content.
   *
   * @param tomlContent the content of a pyproject.toml file
   * @return list of source root paths (relative), empty if none found or on parse error
   */
  public static List<String> extract(String tomlContent) {
    try {
      PyProjectConfig config = TOML_MAPPER.readValue(tomlContent, PyProjectConfig.class);
      return extractFromConfig(config);
    } catch (IOException e) {
      return List.of();
    }
  }

  /**
   * Extracts source root directories from a pyproject.toml File.
   *
   * @param file the pyproject.toml file
   * @return list of source root paths (relative), empty if none found or on parse error
   */
  public static List<String> extract(File file) {
    try {
      return extract(Files.readString(file.toPath()));
    } catch (IOException e) {
      return List.of();
    }
  }

  private static List<String> extractFromConfig(PyProjectConfig config) {
    Set<String> sourceRoots = new LinkedHashSet<>();

    // Check build-system.build-backend for uv_build
    sourceRoots.addAll(extractFromBuildSystem(config.buildSystem()));

    if (config.tool() != null) {
      Tool tool = config.tool();
      sourceRoots.addAll(extractFromSetuptools(tool.setuptools()));
      sourceRoots.addAll(extractFromPoetry(tool.poetry()));
      sourceRoots.addAll(extractFromHatchling(tool.hatch()));
      sourceRoots.addAll(extractFromUvBuild(tool.uv()));
      sourceRoots.addAll(extractFromPdm(tool.pdm()));
      sourceRoots.addAll(extractFromFlit(tool.flit()));
    }

    return new ArrayList<>(sourceRoots);
  }

  // === Build System ===
  // [build-system]
  // build-backend = "uv_build"

  private static List<String> extractFromBuildSystem(@Nullable BuildSystem buildSystem) {
    if (buildSystem == null || buildSystem.buildBackend() == null) {
      return List.of();
    }
    // uv_build auto-detects src/ layout by convention
    if ("uv_build".equals(buildSystem.buildBackend())) {
      return List.of("src");
    }
    return List.of();
  }

  // === Setuptools ===
  // [tool.setuptools.packages.find]
  // where = ["src"]

  private static List<String> extractFromSetuptools(@Nullable Setuptools setuptools) {
    if (setuptools == null || setuptools.packages() == null || setuptools.packages().find() == null) {
      return List.of();
    }
    return setuptools.packages().find().where();
  }

  // === Poetry ===
  // [tool.poetry]
  // packages = [{ include = "mypackage", from = "src" }]

  private static List<String> extractFromPoetry(@Nullable Poetry poetry) {
    if (poetry == null) {
      return List.of();
    }
    return poetry.packages().stream()
      .map(PoetryPackage::from)
      .filter(from -> from != null && !from.isEmpty())
      .distinct()
      .toList();
  }

  // === Hatchling ===
  // [tool.hatch.build.targets.wheel]
  // sources = ["src"]
  // OR packages = ["src/mypackage"]

  private static List<String> extractFromHatchling(@Nullable Hatch hatch) {
    if (hatch == null || hatch.build() == null || hatch.build().targets() == null
      || hatch.build().targets().wheel() == null) {
      return List.of();
    }

    HatchWheel wheel = hatch.build().targets().wheel();

    // Prefer explicit sources
    if (!wheel.sources().isEmpty()) {
      return wheel.sources();
    }

    // Fall back to parsing directory from packages
    if (!wheel.packages().isEmpty()) {
      return wheel.packages().stream()
        .map(PyProjectTomlSourceRoots::extractDirectoryFromPackagePath)
        .filter(dir -> dir != null && !dir.isEmpty())
        .distinct()
        .toList();
    }

    return List.of();
  }

  private static String extractDirectoryFromPackagePath(String packagePath) {
    // Normalize Windows paths to Unix-style
    String normalizedPath = packagePath.replace('\\', '/');
    int slashIndex = normalizedPath.indexOf('/');
    if (slashIndex > 0) {
      return normalizedPath.substring(0, slashIndex);
    }
    return null;
  }

  // === uv_build ===
  // [tool.uv.build-backend]
  // module-root = "src"

  // uv auto-detects src/ layout by convention
  // No explicit configuration needed - returns "src" if uv module is configured

  private static List<String> extractFromUvBuild(@Nullable Uv uv) {
    if (uv == null || uv.buildBackend() == null) {
      return List.of();
    }
    String moduleRoot = uv.buildBackend().moduleRoot();
    if (moduleRoot != null && !moduleRoot.isEmpty()) {
      return List.of(moduleRoot);
    }
    return List.of("src");
  }

  // === PDM ===
  // [tool.pdm]
  // package-dir = "src"

  private static List<String> extractFromPdm(@Nullable Pdm pdm) {
    if (pdm == null) {
      return List.of();
    }
    String packageDir = pdm.packageDir();
    if (packageDir != null && !packageDir.isEmpty()) {
      return List.of(packageDir);
    }
    return List.of();
  }

  // === Flit ===
  // Flit auto-detects src/ layout by convention
  // No explicit configuration needed - returns "src" if flit module is configured

  private static List<String> extractFromFlit(@Nullable Flit flit) {
    // Flit doesn't have explicit source root config
    // It auto-detects src/ layout when the module isn't at project root
    // We return "src" as a hint when flit.module is configured
    if (flit != null && flit.module() != null && flit.module().name() != null) {
      return List.of("src");
    }
    return List.of();
  }

  // === TOML Record Definitions ===

  private record PyProjectConfig(
    @JsonProperty("build-system") @Nullable BuildSystem buildSystem,
    @Nullable Tool tool
  ) {
  }

  private record BuildSystem(
    @JsonProperty("build-backend") @Nullable String buildBackend,
    @JsonSetter(nulls = Nulls.AS_EMPTY) @Nonnull List<String> requires
  ) {
    BuildSystem {
      requires = requires != null ? requires : List.of();
    }
  }

  private record Tool(
    @Nullable Setuptools setuptools,
    @Nullable Poetry poetry,
    @Nullable Hatch hatch,
    @Nullable Uv uv,
    @Nullable Pdm pdm,
    @Nullable Flit flit
  ) {
  }

  // Setuptools records
  private record Setuptools(
    @Nullable SetuptoolsPackages packages
  ) {
  }

  private record SetuptoolsPackages(
    @Nullable SetuptoolsFind find
  ) {
  }

  private record SetuptoolsFind(
    @JsonSetter(nulls = Nulls.AS_EMPTY) @Nonnull List<String> where
  ) {
    SetuptoolsFind {
      where = where != null ? where : List.of();
    }
  }

  // Poetry records
  private record Poetry(
    @JsonSetter(nulls = Nulls.AS_EMPTY) @Nonnull List<PoetryPackage> packages,
    @JsonSetter(nulls = Nulls.AS_EMPTY) @Nonnull Map<String, String> dependencies
  ) {
    Poetry {
      packages = packages != null ? packages : List.of();
      dependencies = dependencies != null ? dependencies : Map.of();
    }
  }

  private record PoetryPackage(
    @Nullable String include,
    @Nullable String from
  ) {
  }

  // Hatchling records
  private record Hatch(
    @Nullable HatchBuild build
  ) {
  }

  private record HatchBuild(
    @Nullable HatchTargets targets
  ) {
  }

  private record HatchTargets(
    @Nullable HatchWheel wheel
  ) {
  }

  private record HatchWheel(
    @JsonSetter(nulls = Nulls.AS_EMPTY) @Nonnull List<String> sources,
    @JsonSetter(nulls = Nulls.AS_EMPTY) @Nonnull List<String> packages
  ) {
    HatchWheel {
      sources = sources != null ? sources : List.of();
      packages = packages != null ? packages : List.of();
    }
  }

  // uv_build records
  private record Uv(
    @JsonProperty("build-backend") @Nullable UvBuildBackend buildBackend
  ) {
  }

  private record UvBuildBackend(
    @JsonProperty("module-root") @Nullable String moduleRoot
  ) {
  }

  // PDM records
  private record Pdm(
    @JsonProperty("package-dir") @Nullable String packageDir
  ) {
  }

  // Flit records
  private record Flit(
    @Nullable FlitModule module
  ) {
  }

  private record FlitModule(
    @Nullable String name
  ) {
  }
}

