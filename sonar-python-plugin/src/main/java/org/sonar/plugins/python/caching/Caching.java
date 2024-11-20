/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.plugins.python.caching;

import com.google.protobuf.InvalidProtocolBufferException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.python.index.Descriptor;
import org.sonar.python.types.protobuf.DescriptorsProtos;

import static org.sonar.python.index.DescriptorsToProtobuf.fromProtobuf;
import static org.sonar.python.index.DescriptorsToProtobuf.toProtobufModuleDescriptor;

public class Caching {

  private final CacheContext cacheContext;

  public static final String IMPORTS_MAP_CACHE_KEY_PREFIX = "python:imports:";
  public static final String PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX = "python:descriptors:";
  public static final String PROJECT_FILES_KEY = "python:files";
  public static final String CONTENT_HASHES_KEY = "python:content_hashes:";
  public static final String TYPESHED_MODULES_KEY = "python:typeshed_modules";
  public static final String CACHE_VERSION_KEY = "python:cache_version";
  public static final String CPD_TOKENS_CACHE_KEY_PREFIX = "python:cpd:data:";
  public static final String CPD_TOKENS_STRING_TABLE_KEY_PREFIX = "python:cpd:stringTable:";

  private static final Logger LOG = LoggerFactory.getLogger(Caching.class);

  public final String cacheVersion;

  public Caching(CacheContext cacheContext, String cacheVersion) {
    this.cacheContext = cacheContext;
    this.cacheVersion = cacheVersion;
  }

  public void writeImportsMapEntry(String fileKey, Set<String> imports) {
    byte[] importData = String.join(";", imports).getBytes(StandardCharsets.UTF_8);
    String cacheKey = importsMapCacheKey(fileKey);
    cacheContext.getWriteCache().write(cacheKey, importData);
  }

  public void writeFileContentHash(String fileKey, byte[] hash) {
    String cacheKey = fileContentHashCacheKey(fileKey);
    cacheContext.getWriteCache().write(cacheKey, hash);
  }

  public void writeFilesList(List<String> mainFiles) {
    byte[] projectFiles = String.join(";", mainFiles).getBytes(StandardCharsets.UTF_8);
    cacheContext.getWriteCache().write(PROJECT_FILES_KEY, projectFiles);
  }

  public void writeTypeshedModules(Set<String> stubModules) {
    byte[] stubModulesBytes = String.join(";", stubModules).getBytes(StandardCharsets.UTF_8);
    cacheContext.getWriteCache().write(TYPESHED_MODULES_KEY, stubModulesBytes);
  }

  public void writeCacheVersion() {
    cacheContext.getWriteCache().write(CACHE_VERSION_KEY, cacheVersion.getBytes(StandardCharsets.UTF_8));
  }

  public void writeProjectLevelSymbolTableEntry(String fileKey, Set<Descriptor> descriptors) {
    String cacheKey = projectSymbolTableCacheKey(fileKey);
    cacheContext.getWriteCache().write(cacheKey, toProtobufModuleDescriptor(descriptors).toByteArray());
  }

  public void copyFromPrevious(String fileKey) {
    cacheContext.getWriteCache().copyFromPrevious(importsMapCacheKey(fileKey));
    cacheContext.getWriteCache().copyFromPrevious(projectSymbolTableCacheKey(fileKey));
    cacheContext.getWriteCache().copyFromPrevious(fileContentHashCacheKey(fileKey));
  }

  @CheckForNull
  public Set<Descriptor> readProjectLevelSymbolTableEntry(String fileKey) {
    String key = projectSymbolTableCacheKey(fileKey);
    if (cacheContext.getReadCache().contains(key)) {
      byte[] bytes = cacheContext.getReadCache().readBytes(key);
      if (bytes != null) {
        try {
          return fromProtobuf(DescriptorsProtos.ModuleDescriptor.parseFrom(bytes));
        } catch (InvalidProtocolBufferException e) {
          LOG.debug("Failed to deserialize project level symbol table entry for module: \"{}\"", fileKey);
        }
      }
    }
    return null;
  }

  @CheckForNull
  public Set<String> readImportMapEntry(String fileKey) {
    String cacheKey = importsMapCacheKey(fileKey);
    byte[] bytes = cacheContext.getReadCache().readBytes(cacheKey);
    if (bytes != null) {
      return new HashSet<>(Arrays.asList(new String(bytes, StandardCharsets.UTF_8).split(";")));
    }
    return null;
  }

  public byte[] readFileContentHash(String fileKey) {
    String cacheKey = fileContentHashCacheKey(fileKey);
    return cacheContext.getReadCache().readBytes(cacheKey);
  }

  public Set<String> readFilesList() {
    return readSet(PROJECT_FILES_KEY);
  }

  public Set<String> readTypeshedModules() {
    return readSet(TYPESHED_MODULES_KEY);
  }

  private Set<String> readSet(String cacheKey) {
    byte[] bytes = cacheContext.getReadCache().readBytes(cacheKey);
    if (bytes != null) {
      return new HashSet<>(Arrays.asList(new String(bytes, StandardCharsets.UTF_8).split(";")));
    }
    return Collections.emptySet();
  }

  public boolean isCacheVersionUpToDate() {
    byte[] bytes = cacheContext.getReadCache().readBytes(CACHE_VERSION_KEY);
    if (bytes != null) {
      String retrievedVersion = new String(bytes, StandardCharsets.UTF_8);
      if (retrievedVersion.equals(cacheVersion)) {
        LOG.debug("Cache version still up to date: \"{}\".", cacheVersion);
        return true;
      }
      LOG.info("The cache version has changed since the previous analysis, cached data will not be used during this analysis." +
        " Retrieved: \"{}\". Current version: \"{}\".", retrievedVersion, cacheVersion);
    }
    return false;
  }

  public boolean isCacheEnabled() {
    return cacheContext.isCacheEnabled();
  }

  public CacheContext cacheContext() {
    return cacheContext;
  }

  public static String importsMapCacheKey(String key) {
    return IMPORTS_MAP_CACHE_KEY_PREFIX + key.replace('\\', '/');
  }

  public static String projectSymbolTableCacheKey(String key) {
    return PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + key.replace('\\', '/');
  }

  public static  String fileContentHashCacheKey(String key) {
    return CONTENT_HASHES_KEY + key.replace('\\', '/');
  }
}
