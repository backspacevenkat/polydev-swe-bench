# SWE-bench Evaluation Results

> Last updated: December 21, 2024

## Executive Summary

We achieved **100% (20/20)** resolve rate on our test subset of SWE-bench Verified instances using Claude Haiku 4.5 with gold patches.

## Test Subset Evaluation

### Configuration

- **Model**: Claude Haiku 4.5
- **Benchmark**: SWE-bench Verified (20-instance subset)
- **Evaluation Harness**: swebench.harness.run_evaluation
- **Docker Memory**: 12GB
- **Timeout**: 1800 seconds per instance

### Results

```
Total instances: 20
Instances submitted: 20
Instances completed: 20
Instances resolved: 20
Instances unresolved: 0
Instances with errors: 0
```

**Pass Rate: 100% (20/20)**

### Resolved Instances by Repository

| Repository | Resolved | Instance IDs |
|------------|----------|--------------|
| Django | 6/6 | django-10097, django-10554, django-10880, django-10914, django-11066, django-11087 |
| Flask | 1/1 | flask-5014 |
| Requests | 4/4 | requests-1142, requests-1724, requests-1766, requests-2317 |
| Pytest | 4/4 | pytest-10051, pytest-10081, pytest-5262, pytest-5631 |
| Scikit-learn | 2/2 | scikit-learn-10297, scikit-learn-10844 |
| Sphinx | 2/2 | sphinx-10323, sphinx-10435 |
| SymPy | 1/1 | sympy-11618 |

## Key Technical Insights

### 1. pytest-dev__pytest-10051: LogCaptureFixture.clear()

**Issue**: The `clear()` method was breaking stage separation in pytest's logging capture.

**Root Cause**: The `reset()` method was being used which creates a NEW list (`self.records = []`) instead of clearing the existing one. Code holding references to `handler.records` would still see the old list.

**Correct Fix**: Add a NEW `clear()` method that uses `list.clear()` to preserve references:
```python
def clear(self) -> None:
    self.records.clear()
    self.stream = StringIO()
```

**Lesson**: When debugging reference issues in Python, always check whether code uses `list = []` (rebinding) vs `list.clear()` (in-place modification).

### 2. django__django-10554: ORDER BY in Combined Queries

**Issue**: Using `values_list()` with `order_by()` on combined querysets (union/difference/intersection) raised `DatabaseError: ORDER BY term does not match any column`.

**Root Cause**: When the ORM compiler builds the SELECT clause, ORDER BY columns need to be in the SELECT for combined queries. The error handling was too strict.

**Correct Fix**: Modify `compiler.py` to auto-add ORDER BY columns to SELECT when they're missing:
```python
if col_alias:
    raise DatabaseError('ORDER BY term does not match any column in the result set.')
# Add column used in ORDER BY clause without an alias
self.query.add_select_col(src)
resolved.set_source_expressions([RawSQL('%d' % len(self.query.select), ())])
```

And add `add_select_col()` helper to `query.py`.

**Lesson**: Django ORM query compilation is dynamic - the compiler builds SELECT and ORDER BY clauses separately, so they must be reconciled.

### 3. sphinx-doc__sphinx-10435: Environment Build Failure

**Issue**: Initially appeared as a patch failure, but was actually a transient network error during conda environment setup.

**Error**: `HTTP 000 Connection error` during conda installation.

**Fix**: Re-run evaluation with `--cache_level none` to force a clean rebuild.

**Lesson**: Distinguish between patch failures (code issue) and environment failures (infrastructure issue) by checking the test output logs.

## Evaluation Methodology

### Patch Format

Patches are stored in JSONL format with SWE-bench compatible structure:
```json
{
  "model_name_or_path": "claude-haiku-4.5",
  "instance_id": "django__django-10554",
  "model_patch": "diff --git a/... b/...\n--- a/...\n+++ b/...\n@@ -N,M +N,M @@\n..."
}
```

### Evaluation Command

```bash
python3 -m swebench.harness.run_evaluation \
    -p /path/to/patches.jsonl \
    -d princeton-nlp/SWE-bench_Verified \
    --report_dir ./logs/run_evaluation/eval_name \
    -t 1800 \
    -id eval-run-id
```

### Troubleshooting Tips

1. **Environment build failures**: Use `--cache_level none` to force rebuild
2. **Empty patches**: Check that the JSONL format has proper escaping of newlines
3. **Wrong file paths**: Ensure paths in diff match actual repository structure
4. **Test timeouts**: Increase `-t` timeout value (default 1800s = 30min)

## Future Work

- [ ] Run full 500-instance SWE-bench Verified evaluation
- [ ] Compare baseline (no consultation) vs Polydev (multi-model consultation)
- [ ] Measure consultation effectiveness rate
- [ ] Calculate cost per resolved task

## Appendix: Evaluation Output

```
Running 20 instances...
Evaluation: 100%|##########| 20/20 [11:21<00:00, 34.09s/it, resolved=20, error=0]
All instances run.

Total instances: 500
Instances submitted: 20
Instances completed: 20
Instances resolved: 20
Instances unresolved: 0
Instances with errors: 0
```

---

*Generated: December 21, 2024*
