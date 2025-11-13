---
description: Define and enforce the testing process. Separate planning from implementation, enforce approvals, and validate correctness using the project's standard test runner.
auto_execution_mode: 1
---

# Testing Workflow

Standalone Capability: This workflow can run independently without requiring prior phase outputs.

## Phase Input (optional)

If Execution output exists, import:
- Plan ID or Stub ID: Use for traceability in test documentation
- Step label: Understand what was implemented
- Files touched: Focus test coverage on these areas
- Acceptance criteria status: Design tests to verify these
- Negative-first gap list: Ensure tests explicitly cover what was NOT implemented
- Validation outcome: If build failed, do not proceed with testing

If no Execution output exists, proceed with standalone test planning based on user request.

## Act as the Test Planner Agent (no code)

Purpose: Define WHAT needs to be tested, independent of testing framework.

Based on imported Execution output (or user request), produce a platform-agnostic test specification:

Test Categories:
- Unit Tests: Test isolated functions, classes, or components
- Integration Tests: Test interaction between modules or services
- E2E Tests (if applicable): Test complete user workflows

For each test scenario, specify:
- Test Scenario: Plain language description of what behavior to verify
- Test Category: Unit/Integration/E2E
- Test Cases: List specific inputs, expected outputs, and edge cases
- Priority: Must-test vs Should-test
- Risk Coverage: Which items from negative-first gap list this addresses

Coverage Requirements:
- What must be tested (from acceptance criteria)
- What should be tested (edge cases, error handling)
- What can be deferred (low-risk areas)

Do not include framework-specific details (no syntax, imports, or file paths).

Stop here and wait for approval.

## Act as the Framework Analyzer Agent (after test plan approval)

Purpose: Prove understanding of the project's testing setup before implementation.

Produce a Testing Framework Report by analyzing the project:

Framework Identification:
- Primary testing framework: [detect from package.json, config files]
- Test runner: [Jest/Karma/Jasmine/Mocha/pytest/JUnit/etc.]
- Assertion library: [if different from framework]
- Additional testing tools: [mocking libraries, testing utilities]

Project Testing Patterns (by examining existing test files):
- Test file location: [pattern like src/**/*.spec.ts, tests/, __tests__/]
- Naming convention: [*.spec.ts, *.test.ts, test_*.py, Test*.java]
- Test structure pattern: [describe/it, test functions, class-based]
- Common imports: [what existing tests import]

Existing Test Examples:
- Show 2-3 code snippets from existing tests in this project
- Demonstrate: file structure, imports, test syntax, assertions, mocking patterns

Configuration Discovery:
- Test config file: [karma.conf.js, jest.config.js, angular.json, pytest.ini]
- Test command: [from package.json scripts or documentation]
- Coverage tool: [istanbul, nyc, coverage.py, etc.]

Dependencies Check:
- Testing-related packages: [list from package.json with versions]

Stop here and present this report.

Explicitly ask: "Does this correctly reflect your testing setup? Should I proceed with test implementation?"

Wait for explicit approval before moving to implementation.

## Act as the Test Implementer Agent (after framework validation approval)

Purpose: Implement tests using the validated framework understanding.

Implementation Process:
- Use the exact patterns shown in the Framework Report
- Follow project's naming and file structure conventions
- Implement one test at a time
- Run each test after implementation
- Report results immediately

For each test:
1. Create/modify test file following project convention
2. Implement test case using validated framework syntax
3. Run the test using validated test command
4. Report: Pass/Fail with details
5. If failing: propose minimal corrective step
6. Wait for approval before moving to next test

Framework Compliance Self-Check (before running):
- File naming matches convention: [check]
- Import statements match existing tests: [check]
- Test structure matches project pattern: [check]
- No unsupported framework features used: [check]

Do not implement production code changes without explicit approval.

## Output Manifest (upon completion)

Upon successful test implementation and execution, provide the following outputs:

Test Traceability:
- Plan ID or Stub ID: Link back to implementation
- Test Coverage Summary: What was tested

Test Results:
- Total tests: [count]
- Passed: [count]
- Failed: [count with details]
- Skipped: [count with reasons]

Test Files:
- Created/modified files: [list with paths]
- Test command used: [exact command]

Coverage Analysis:
- Acceptance criteria coverage: Which criteria now have test coverage
- Gap coverage: Which negative-first gaps are now tested
- Remaining gaps: What still needs testing

Note: These outputs can be used for documentation or future test maintenance.
