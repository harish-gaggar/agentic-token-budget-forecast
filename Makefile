PAPER = paper
PY = .venv/bin/python

# Extra TeXLive packages the small docker image needs for this paper.
TL_EXTRA = environ caption float nicefrac units pgf microtype xkeyval \
           multirow enumitem placeins

# Docker image used by the `paper` target. Override with PAPER_IMAGE=...
PAPER_IMAGE ?= texlive/texlive:latest-small

.PHONY: all paper paper-local paper-txt experiments production microbench \
        generic-agents multiseed stagewise calibration stress prod-extract \
        tabf-replay reviewer-evidence verify view clean distclean venv

# Public reproducibility path (figures + CSVs). Paper PDF is checked in.
all: experiments production

# Bundle the reviewer-requested experiments (Sections 6.5-6.8 and 7.5)
# so they can be re-run with a single make target. Each underlying
# script also has its own target if you only need one piece.
reviewer-evidence: multiseed stagewise calibration stress tabf-replay

# Default `make paper` runs the LaTeX build inside docker so contributors
# do not need a local LaTeX install. It installs the missing packages
# the small TeXLive image does not ship with, then runs the standard
# pdflatex / bibtex / pdflatex / pdflatex cycle, and finally regenerates
# the plain-text version of the paper.

paper: $(PAPER).pdf paper-txt

$(PAPER).pdf: $(PAPER).tex $(PAPER).bib neurips_2024.sty \
              results/fig_w20r.pdf \
              results/fig_production_savings.pdf \
              results/fig_production_costs.pdf \
              results/fig_generic_agents.pdf
	docker run --rm --platform linux/amd64 -v "$(CURDIR)":/data -w /data \
	  $(PAPER_IMAGE) bash -c '\
	    tlmgr install $(TL_EXTRA) >/dev/null 2>&1 || true; \
	    pdflatex -interaction=nonstopmode $(PAPER).tex && \
	    bibtex $(PAPER) && \
	    pdflatex -interaction=nonstopmode $(PAPER).tex && \
	    pdflatex -interaction=nonstopmode $(PAPER).tex'

# `make paper-local` skips docker and runs pdflatex on the host. Use
# this if you have MacTeX / TeX Live installed locally and want a
# faster rebuild.

paper-local: $(PAPER).tex $(PAPER).bib neurips_2024.sty \
             results/fig_w20r.pdf \
             results/fig_production_savings.pdf \
             results/fig_production_costs.pdf \
             results/fig_generic_agents.pdf
	pdflatex -interaction=nonstopmode $(PAPER).tex
	bibtex $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex
	$(MAKE) paper-txt

paper-txt: $(PAPER).txt

$(PAPER).txt: $(PAPER).tex experiments/build_paper_txt.py
	$(PY) -m experiments.build_paper_txt

# --- Synthetic benchmark (TABF / Sections 5-6) --------------------------------

results/fig_w20r.pdf: experiments/run_all.py tabf/*.py benchmark/*.py
	MPLCONFIGDIR=/tmp/mplcache $(PY) -m experiments.run_all --out results

experiments:
	MPLCONFIGDIR=/tmp/mplcache $(PY) -m experiments.run_all --out results

# --- Production case study (ContextOptimizer / Sections 7-8) ------------------
# Reads results/production_*.csv + results/optimizer_microbench*.csv
# (checked in, real measurements) and regenerates the three production
# figures. Re-run after editing any of the source CSVs.

results/fig_production_savings.pdf \
results/fig_production_costs.pdf \
results/fig_layer_attribution.pdf: \
        experiments/plot_production.py \
        results/production_token_savings_summary.csv \
        results/production_token_savings_samples.csv \
        results/production_cost_projections.csv \
        results/optimizer_microbench.csv \
        results/optimizer_microbench_layers.csv
	MPLCONFIGDIR=/tmp/mplcache $(PY) -m experiments.plot_production --out results

production:
	MPLCONFIGDIR=/tmp/mplcache $(PY) -m experiments.plot_production --out results

# --- Optimiser microbench (Section 8.5) --------------------------------------
# Regenerates results/optimizer_microbench{,_layers}.csv from a 100-run
# microbench of the balanced() optimiser preset inside the production
# container. Requires a running container that has
# agent.context_optimizer + langchain_core installed. See the
# experiments/optimizer_microbench.py module docstring for the exact
# docker exec commands.

microbench:
	@echo "Run experiments/optimizer_microbench.py inside the agent"
	@echo "container; see its module docstring for the exact commands."
	@echo "The script writes optimizer_microbench{,_layers}.csv that this"
	@echo "Makefile expects at results/."

# --- Generic-agent persona benchmark (Section 8.8) ---------------------------
# Regenerates results/generic_agents{,_layers}.csv from a 100-run benchmark
# (4 personas x 25 seeds) of the balanced() preset against synthetic
# message lists shaped like four different agent types. Like the
# microbench, this must run inside the agent container that has
# agent.context_optimizer + langchain_core installed. See the
# experiments/generic_agents.py module docstring for the exact docker
# commands. The figure target below regenerates the PDF from the CSVs.

results/fig_generic_agents.pdf: \
        experiments/plot_generic_agents.py \
        results/generic_agents.csv \
        results/generic_agents_layers.csv
	MPLCONFIGDIR=/tmp/mplcache $(PY) -m experiments.plot_generic_agents --out results

generic-agents:
	@echo "Run experiments/generic_agents.py inside the agent"
	@echo "container; see its module docstring for the exact commands."
	@echo "The script writes generic_agents{,_layers}.csv that the"
	@echo "results/fig_generic_agents.pdf target reads."

# --- Reviewer-evidence experiments (Sections 6.5-6.8 and 7.5) ----------------
# Each target re-runs one of the five experiments added in response
# to reviewer feedback. They all write into results/ and are read by
# the matching tables in paper.tex.

multiseed:
	MPLCONFIGDIR=/tmp/mplcache $(PY) -m experiments.multiseed --out results --seeds 5

stagewise:
	MPLCONFIGDIR=/tmp/mplcache $(PY) -m experiments.stagewise --out results

calibration:
	MPLCONFIGDIR=/tmp/mplcache $(PY) -m experiments.calibration --out results

stress:
	MPLCONFIGDIR=/tmp/mplcache $(PY) -m experiments.stress_test --out results

# Pulls the latest production turn-level + quality-eval data from
# the BigQuery agent-eval pipeline. Requires bq CLI + gcloud auth
# pointed at prod-ck-analytics-data-99. Overwrites the existing
# results/production_*.csv snapshots.
prod-extract:
	$(PY) -m experiments.prod_extract --out results

# Offline replay of TABF on the latest production turn extract.
# Reads results/production_bq_turns.csv (produced by prod-extract).
tabf-replay:
	MPLCONFIGDIR=/tmp/mplcache $(PY) -m experiments.tabf_replay \
	  --out results --prod-csv results/production_bq_turns.csv

stagewise-gbt:
	MPLCONFIGDIR=/tmp/mplcache $(PY) -m experiments.stagewise_gbt_eval --out results

real-trace:
	$(PY) -m experiments.real_trace_eval --out results

quality-outcomes:
	$(PY) -m experiments.quality_outcomes --out results

compression-baseline:
	$(PY) -m experiments.compression_baseline --out results

oracle-labels:
	$(PY) -m experiments.oracle_labels_eval --out results

reviewer-evidence: multiseed stagewise calibration stress tabf-replay \
	stagewise-gbt real-trace quality-outcomes compression-baseline

# Stagewise + calibration figures are paper dependencies; re-make
# them whenever their generating scripts change.
results/fig_stagewise.pdf: experiments/stagewise.py tabf/*.py benchmark/*.py
	MPLCONFIGDIR=/tmp/mplcache $(PY) -m experiments.stagewise --out results

results/fig_calibration_reliability.pdf: experiments/calibration.py tabf/*.py benchmark/*.py
	MPLCONFIGDIR=/tmp/mplcache $(PY) -m experiments.calibration --out results

# End-to-end reproducibility: code, experiments, figures (no LaTeX; no git).
verify:
	@bash scripts/verify.sh

venv:
	@command -v python3.12 >/dev/null 2>&1 && PY=python3.12 || \
	 command -v python3.11 >/dev/null 2>&1 && PY=python3.11 || \
	 command -v python3.10 >/dev/null 2>&1 && PY=python3.10 || PY=python3; \
	 $$PY -m venv .venv
	PIP_INDEX_URL=$${PIP_INDEX_URL:-https://pypi.org/simple} \
	  $(PY) -m pip install --index-url $$PIP_INDEX_URL -r requirements.txt

view: $(PAPER).pdf
	open $(PAPER).pdf

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log \
	      $(PAPER).out $(PAPER).toc $(PAPER).synctex.gz \
	      $(PAPER).fls $(PAPER).fdb_latexmk

distclean: clean
	rm -f $(PAPER).pdf
	rm -rf results/
