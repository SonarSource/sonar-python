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
package org.sonar.python.caching;


import com.google.protobuf.InvalidProtocolBufferException;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.junit.Test;
import org.sonar.plugins.python.api.caching.PythonReadCache;
import org.sonar.plugins.python.api.caching.PythonWriteCache;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.DescriptorUtils;
import org.sonar.python.semantic.ProjectLevelSymbolTable;

import static org.sonar.python.PythonTestUtils.parseWithoutSymbols;
import static org.sonar.python.PythonTestUtils.pythonFile;
import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.caching.Caching.PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX;

public class CachingTest {

  @Test
  public void testwriteProjectLevelSymbolTableEntry() throws InvalidProtocolBufferException {
    TestWriteCache writeCache = new TestWriteCache();
    PythonWriteCache pythonWriteCache = new PythonWriteCacheImpl(writeCache);
    TestReadCache readCache = new TestReadCache();
    PythonReadCache pythonReadCache = new PythonReadCacheImpl(readCache);
    CacheContextImpl cacheContext = new CacheContextImpl(true, pythonWriteCache, pythonReadCache);

    FileInput tree = parseWithoutSymbols(
      "class A: ...",
      "class B(A): ..."
    );
    ProjectLevelSymbolTable projectLevelSymbolTable = new ProjectLevelSymbolTable();
    projectLevelSymbolTable.addModule(tree, "", pythonFile("mod.py"));

    Caching caching = new Caching(cacheContext);
    Set<Descriptor> initialDescriptors = projectLevelSymbolTable.descriptorsForModule("mod");
    caching.writeProjectLevelSymbolTableEntry("mod", initialDescriptors);
    Map<String, byte[]> data = writeCache.getData();
    String cacheKey = PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + "mod";
    Set<Descriptor> retrievedDescriptors = DescriptorUtils.deserializeProtobufDescriptors(data.get(cacheKey));
    assertThat(retrievedDescriptors).containsExactlyInAnyOrderElementsOf(initialDescriptors);
  }


  @Test
  public void testreadProjectLevelSymbolTableEntry() throws InvalidProtocolBufferException {
    TestWriteCache writeCache = new TestWriteCache();
    PythonWriteCache pythonWriteCache = new PythonWriteCacheImpl(writeCache);
    TestReadCache readCache = new TestReadCache();
    PythonReadCache pythonReadCache = new PythonReadCacheImpl(readCache);
    CacheContextImpl cacheContext = new CacheContextImpl(true, pythonWriteCache, pythonReadCache);

    FileInput tree = parseWithoutSymbols(
      "class A: ...",
      "class B(A): ..."
    );
    ProjectLevelSymbolTable projectLevelSymbolTable = new ProjectLevelSymbolTable();
    projectLevelSymbolTable.addModule(tree, "", pythonFile("mod.py"));

    Caching caching = new Caching(cacheContext);
    Set<Descriptor> initialDescriptors = projectLevelSymbolTable.descriptorsForModule("mod");
    String cacheKey = PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + "mod";
    readCache.put(cacheKey, Caching.moduleDescriptor(initialDescriptors).toByteArray());
    Optional<Set<Descriptor>> retrievedDescriptorsOptional = caching.readProjectLevelSymbolTableEntry(cacheKey);
    assertThat(retrievedDescriptorsOptional).isPresent();
    assertThat(retrievedDescriptorsOptional.get()).containsExactlyInAnyOrderElementsOf(initialDescriptors);
  }
}
