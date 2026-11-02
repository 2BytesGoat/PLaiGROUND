# repo-brief migration (2026-11-02)

Retired when PLaiGROUND adopted `repo-brief`, vendored at
`tools/vendor/repo-brief`. Kept for reference; nothing here runs.

| File | What it did |
|---|---|
| `repo_index.py` | Built `docs/code-map.md` and a local `.cache/repo-index.json`. |
| `context.py` | Ranked that index for a task query, including `runs/`. |
| `test_repo_search.py` | Tests for both tools. |
| `code-map.md` | The last generated map. |

The current map is `.repo-brief/map.md` (`make index`), context is
`make context`, and the reason for the swap is in
[`docs/project/decisions.md`](../../docs/project/decisions.md).
