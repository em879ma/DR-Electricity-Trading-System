# DR Day-Ahead Decision System
# Pipeline: forecast → scenario generation → day-ahead DR decision → counterfactual market outcome
#
# Key modules:
#   data_download          - OPSD and Ember data download
#   data_loader            - Load and parse OPSD data
#   data_cleaning          - Clean and standardize market panel
#   feature_engineering    - Calendar, lag, rolling, energy-mix features
#   energy_mix             - Ember yearly features + regime analysis
#   elasticity             - Rolling elasticity (feature) + dynamic estimation
#   demand_response        - Baseline demand + behavioral response
#   forecasting            - RF + persistence + rolling-mean baselines
#   calibration            - Linear calibration of RF forecasts
#   scenarios              - Bootstrap scenario generation
#   day_ahead_decision     - Grid-search DR optimization with guardrails
#   counterfactual_price_model - Ridge + RF structural price model
#   evaluation_forecast    - Level 1: forecast accuracy
#   evaluation_scenarios   - Level 2: scenario quality
#   evaluation_decision    - Level 3: decision performance
#   evaluation_mechanism   - Level 4: economic mechanism validity
#   visualization          - 6 final figures (01-06)
#   reporting              - final_research_summary.md
