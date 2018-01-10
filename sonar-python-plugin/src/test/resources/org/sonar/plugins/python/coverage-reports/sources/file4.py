# the coverage of this file has been generated using coverage-4.4.2:
# cd sonar-python-plugin/src/test/resources/org/sonar/plugins/python/coverage-reports/sources
# coverage run --branch file4.py
# coverage xml -o ../coverage.4.4.2.xml

def method1(a, b):
  print("Covered 1")
  if a == 2 and b == 1:
    print("Uncovered")
  elif a == 1 and b == 1:
    print("Covered 2")
  else:
    print("Uncovered")

method1(
    1,
    1)
