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
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.Test;
import org.mockito.Mockito;
import org.sonar.api.utils.log.LogTester;
import org.sonar.api.utils.log.LoggerLevel;
import org.sonar.plugins.python.api.caching.PythonReadCache;
import org.sonar.plugins.python.api.caching.PythonWriteCache;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.DescriptorUtils;
import org.sonar.python.semantic.ProjectLevelSymbolTable;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.sonar.python.PythonTestUtils.parseWithoutSymbols;
import static org.sonar.python.PythonTestUtils.pythonFile;
import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.caching.Caching.IMPORTS_MAP_CACHE_KEY_PREFIX;
import static org.sonar.python.caching.Caching.PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX;

public class CachingTest {

  @org.junit.Rule
  public LogTester logTester = new LogTester();


  @Test
  public void writeProjectLevelSymbolTableEntry() throws InvalidProtocolBufferException {
    TestWriteCache writeCache = new TestWriteCache();
    PythonWriteCache pythonWriteCache = new PythonWriteCacheImpl(writeCache);
    TestReadCache readCache = new TestReadCache();
    PythonReadCache pythonReadCache = new PythonReadCacheImpl(readCache);
    CacheContextImpl cacheContext = new CacheContextImpl(true, pythonWriteCache, pythonReadCache);

    FileInput tree = parseWithoutSymbols(
      "class A: ...",
      "class B(A): ...",
      "def foo(): ...",
      "x :int = 42",
      "def bar(): ...",
      "bar = 24"
    );
    ProjectLevelSymbolTable projectLevelSymbolTable = new ProjectLevelSymbolTable();
    projectLevelSymbolTable.addModule(tree, "", pythonFile("mod.py"));

    Caching caching = new Caching(cacheContext);
    Set<Descriptor> initialDescriptors = projectLevelSymbolTable.descriptorsForModule("mod");
    caching.writeProjectLevelSymbolTableEntry("mod", initialDescriptors);
    Map<String, byte[]> data = writeCache.getData();
    String cacheKey = PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + "mod";
    Set<Descriptor> retrievedDescriptors = DescriptorUtils.deserializeProtobufDescriptors(data.get(cacheKey));
    assertThat(cacheContext.isCacheEnabled()).isTrue();
    assertThat(retrievedDescriptors).usingRecursiveFieldByFieldElementComparator().containsExactlyInAnyOrderElementsOf(initialDescriptors);
  }


  @Test
  public void readProjectLevelSymbolTableEntry() {
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
    Set<Descriptor> retrievedDescriptorsOptional = caching.readProjectLevelSymbolTableEntry(cacheKey);
    assertThat(retrievedDescriptorsOptional).isNotNull().usingRecursiveFieldByFieldElementComparator().containsExactlyInAnyOrderElementsOf(initialDescriptors);
  }

  @Test
  public void readProjectLevelSymbolTableMissingEntry() {
    TestWriteCache writeCache = new TestWriteCache();
    TestReadCache readCache = new TestReadCache();
    CacheContextImpl cacheContext = new CacheContextImpl(true, new PythonWriteCacheImpl(writeCache), new PythonReadCacheImpl(readCache));

    Caching caching = new Caching(cacheContext);
    assertThat(caching.readProjectLevelSymbolTableEntry("unknown")).isNull();
  }

  @Test
  public void readProjectLevelSymbolTableIOException() throws IOException {
    TestWriteCache writeCache = new TestWriteCache();
    TestReadCache readCache = new TestReadCache();
    InputStream inputStream = mock(InputStream.class);
    when(inputStream.readAllBytes()).thenThrow(new IOException("Boom!"));
    PythonReadCacheImpl pythonReadCache = Mockito.spy(new PythonReadCacheImpl(readCache));
    readCache.put("key", new byte[0]);
    Mockito.when(pythonReadCache.read("key")).thenReturn(inputStream);

    CacheContextImpl cacheContext = new CacheContextImpl(true, new PythonWriteCacheImpl(writeCache), pythonReadCache);
    Caching caching = new Caching(cacheContext);
    assertThat(caching.readProjectLevelSymbolTableEntry("key")).isNull();
    assertThat(logTester.logs(LoggerLevel.DEBUG)).contains("Unable to read data for key \"key\"");
  }

  @Test
  public void writeImportsMapEntry() {
    TestWriteCache writeCache = new TestWriteCache();
    PythonWriteCache pythonWriteCache = new PythonWriteCacheImpl(writeCache);
    PythonReadCache pythonReadCache = new PythonReadCacheImpl(new TestReadCache());
    CacheContextImpl cacheContext = new CacheContextImpl(true, pythonWriteCache, pythonReadCache);

    Caching caching = new Caching(cacheContext);
    Set<String> imports = Set.of("mod2", "pkg1.mod3", "pkg2.pkg3.mod4");

    String cacheKey = IMPORTS_MAP_CACHE_KEY_PREFIX + "mod";
    caching.writeImportMapEntry("mod", imports);
    Map<String, byte[]> data = writeCache.getData();
    Set<String> retrievedDescriptors = Arrays.stream(new String(data.get(cacheKey), StandardCharsets.UTF_8).split(";")).collect(Collectors.toSet());
    assertThat(retrievedDescriptors).containsExactlyInAnyOrderElementsOf(imports);
  }

  @Test
  public void readImportsMapEntry() {
    TestWriteCache writeCache = new TestWriteCache();
    TestReadCache readCache = new TestReadCache();
    writeCache.bind(readCache);
    PythonWriteCache pythonWriteCache = new PythonWriteCacheImpl(writeCache);
    PythonReadCache pythonReadCache = new PythonReadCacheImpl(readCache);
    CacheContextImpl cacheContext = new CacheContextImpl(true, pythonWriteCache, pythonReadCache);

    Caching caching = new Caching(cacheContext);
    Set<String> imports = Set.of("mod2", "pkg1.mod3", "pkg2.pkg3.mod4");
    String cacheKey = IMPORTS_MAP_CACHE_KEY_PREFIX + "mod";
    readCache.put(cacheKey, String.join(";", imports).getBytes(StandardCharsets.UTF_8));
    assertThat(caching.readImportMapEntry("mod")).containsExactlyInAnyOrderElementsOf(imports);
  }

  @Test
  public void readImportsMissingEntry() {
    TestWriteCache writeCache = new TestWriteCache();
    TestReadCache readCache = new TestReadCache();
    CacheContextImpl cacheContext = new CacheContextImpl(true, new PythonWriteCacheImpl(writeCache), new PythonReadCacheImpl(readCache));

    Caching caching = new Caching(cacheContext);
    assertThat(caching.readImportMapEntry("unknown")).isNull();
  }

  @Test
  public void corruptedDataInCache() {
    TestWriteCache writeCache = new TestWriteCache();
    PythonWriteCache pythonWriteCache = new PythonWriteCacheImpl(writeCache);
    TestReadCache readCache = new TestReadCache();
    PythonReadCache pythonReadCache = new PythonReadCacheImpl(readCache);
    CacheContextImpl cacheContext = new CacheContextImpl(true, pythonWriteCache, pythonReadCache);


    Caching caching = new Caching(cacheContext);
    String cacheKey = PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + "mod";
    readCache.put(cacheKey, new byte[] {42});
    assertThat(caching.readProjectLevelSymbolTableEntry(cacheKey)).isNull();
    assertThat(logTester.logs(LoggerLevel.DEBUG)).contains("Failed to deserialize project level symbol table entry for key: python:descriptors:mod");
  }
}
