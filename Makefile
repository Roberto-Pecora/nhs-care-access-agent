.PHONY: test eval
test:
	uv run pytest
eval:
	uv run nhs-care-access-agent evaluate --taskset evals/seed_tasks.jsonl
