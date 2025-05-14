.PHONY: rebuild-strength-data

# Rebuild strength_sessions.parquet and strength_exercises_log.parquet from raw logs
rebuild-strength-data:
	@echo "Rebuilding strength data from raw logs..."
	@rm -f cultivation/data/strength/processed/strength_sessions.parquet \
	       cultivation/data/strength/processed/strength_exercises_log.parquet
	@for f in cultivation/data/strength/raw/*.md; do \
		if [ "$(basename $$f)" = "strength_log_template.md" ]; then continue; fi; \
		.venv/bin/python -m cultivation.scripts.strength.ingest_yaml_log $$f; \
	done
	@echo "Rebuild complete."
