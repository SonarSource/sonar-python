/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.plugins.python.caching;

import com.google.protobuf.InvalidProtocolBufferException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
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
  public static final String CACHE_VERSION_KEY = "python:cache_version";

  private static final Logger LOG = Loggers.get(Caching.class);

  public final String cacheVersion;

  public Caching(CacheContext cacheContext, String cacheVersion) {
    this.cacheContext = cacheContext;
    this.cacheVersion = cacheVersion;
  }

  public void writeImportsMapEntry(String moduleFqn, Set<String> imports) {
    byte[] importData = String.join(";", imports).getBytes(StandardCharsets.UTF_8);
    String cacheKey = IMPORTS_MAP_CACHE_KEY_PREFIX + moduleFqn;
    cacheContext.getWriteCache().write(cacheKey, importData);
  }

  public void writeFilesList(List<String> mainFiles) {
    byte[] projectFiles = String.join(";", mainFiles).getBytes(StandardCharsets.UTF_8);
    cacheContext.getWriteCache().write(PROJECT_FILES_KEY, projectFiles);
  }

  public void writeCacheVersion() {
    cacheContext.getWriteCache().write(CACHE_VERSION_KEY, cacheVersion.getBytes(StandardCharsets.UTF_8));
  }

  public void writeProjectLevelSymbolTableEntry(String moduleFqn, Set<Descriptor> descriptors) {
    String cacheKey = PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + moduleFqn;
    cacheContext.getWriteCache().write(cacheKey, toProtobufModuleDescriptor(descriptors).toByteArray());
  }

  public void copyFromPrevious(String moduleFqn) {
    cacheContext.getWriteCache().copyFromPrevious(IMPORTS_MAP_CACHE_KEY_PREFIX + moduleFqn);
    cacheContext.getWriteCache().copyFromPrevious(PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + moduleFqn);
  }

  @CheckForNull
  public Set<Descriptor> readProjectLevelSymbolTableEntry(String module) {
    String key = PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + module;
    if (cacheContext.getReadCache().contains(key)) {
      byte[] bytes = cacheContext.getReadCache().readBytes(key);
      if (bytes != null) {
        try {
          return fromProtobuf(DescriptorsProtos.ModuleDescriptor.parseFrom(bytes));
        } catch (InvalidProtocolBufferException e) {
          LOG.debug("Failed to deserialize project level symbol table entry for module: \"{}\"", module);
        }
      }
    }
    return null;
  }

  @CheckForNull
  public Set<String> readImportMapEntry(String moduleFqn) {
    String cacheKey = IMPORTS_MAP_CACHE_KEY_PREFIX + moduleFqn;
    byte[] bytes = cacheContext.getReadCache().readBytes(cacheKey);
    if (bytes != null) {
      return new HashSet<>(Arrays.asList(new String(bytes, StandardCharsets.UTF_8).split(";")));
    }
    return null;
  }

  public Set<String> readFilesList() {
    byte[] bytes = cacheContext.getReadCache().readBytes(PROJECT_FILES_KEY);
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
        String.format(" Retrieved: \"%s\". Current version: \"%s\".", retrievedVersion, cacheVersion));
    }
    return false;
  }

  public boolean isCacheEnabled() {
    return cacheContext.isCacheEnabled();
  }

  public CacheContext cacheContext() {
    return cacheContext;
  }
}
