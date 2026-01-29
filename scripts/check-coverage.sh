#!/bin/bash
# Coverage check script for Python checks
# Usage: ./scripts/check-coverage.sh <TestClassName>
# Example: ./scripts/check-coverage.sh FastAPIFileUploadFormCheckTest

set -e

if [ -z "$1" ]; then
    echo "Error: Test class name required"
    echo "Usage: $0 <TestClassName>"
    echo "Example: $0 FastAPIFileUploadFormCheckTest"
    exit 1
fi

TEST_CLASS="$1"
CHECK_CLASS="${TEST_CLASS%Test}"  # Remove 'Test' suffix to get check class name

echo "Running coverage for $TEST_CLASS..."
echo "Check class: $CHECK_CLASS"
echo ""

# Run tests with JaCoCo coverage
mvn org.jacoco:jacoco-maven-plugin:0.8.12:prepare-agent test \
    org.jacoco:jacoco-maven-plugin:0.8.12:report \
    -Dtest="$TEST_CLASS" \
    -pl python-checks \
    -DskipTypeshed \
    -q

# Parse and display coverage results
echo ""
echo "Coverage Results:"
echo "================"
awk -F, -v class="$CHECK_CLASS" '
    $3 == class {
        line_cov = $9
        line_total = $8 + $9
        line_pct = (line_total > 0) ? (line_cov / line_total * 100) : 0

        branch_cov = $7
        branch_total = $6 + $7
        branch_pct = (branch_total > 0) ? (branch_cov / branch_total * 100) : 0

        printf "Lines:    %d/%d (%.1f%%)\n", line_cov, line_total, line_pct
        printf "Branches: %d/%d (%.1f%%)\n", branch_cov, branch_total, branch_pct

        if (line_pct == 100 && branch_pct == 100) {
            printf "\n✓ Perfect coverage achieved!\n"
        } else {
            printf "\nUncovered:\n"
            printf "  Lines:    %d\n", $8
            printf "  Branches: %d\n", $6
        }
    }
' python-checks/target/site/jacoco/jacoco.csv

# Display location of detailed report
echo ""
echo "Detailed report: python-checks/target/site/jacoco/org.sonar.python.checks/${CHECK_CLASS}.html"
