# 2. Data cleaning

We used the raw Paraguay electricity consumption dataset, containing 1,899,312 rows for 55 feeders from January 2017 to December 2020. We plotted each feeder separately to inspect its time series without masking missing measurements through aggregation.

The pandas `.isnull()` check identified 50,365 missing consumption values and no missing timestamps, substation identifiers, or feeder identifiers in existing rows. This check cannot detect rows that are absent. Reindexing each feeder to the full hourly study period revealed another 29,208 absent timestamps, giving 79,573 missing readings out of 1,928,520 expected readings (4.13%). Following the supplied script, this assumes each feeder should have observations throughout the study period; absent hours could instead reflect periods when a feeder was not operating, which requires confirmation from dataset documentation.

## Method and justification

For each missing reading, we calculated the mean of available original readings from the same feeder at the same hour and weekday, from up to six weeks before and six weeks after the gap. Each estimate therefore used between one and twelve observations. We did not reuse imputed values as inputs for other estimates.

This method follows the weekly averaging idea in the supplied `data-imputation.py` script. It accounts for possible hourly and weekly consumption patterns and differences between feeders, which a single overall mean or median would ignore. The six-week window follows the supplied script and is a heuristic, not a parameter shown to be optimal. Unlike interpolation over long gaps, it uses comparable weekly time slots rather than assuming a straight-line change. Zero filling would incorrectly treat missing measurements as zero consumption.

The procedure filled 77,372 readings. The remaining 2,201 had no available weekly donors in the window. We retained these as missing in the audit file and omitted only these individual readings from the cleaned export. We did not discard entire days or remove observations from other feeders. This avoids inventing unsupported estimates, although dropping unresolved readings can introduce bias if missingness is systematic. The cleaned export has 1,926,319 rows; it is no longer a complete hourly grid where gaps remain.

All originally observed consumption values were preserved. Outlier treatment belongs to Task 3 and was not applied here. The audit file records whether a row existed in the raw CSV, whether its consumption was imputed, and how many weekly donors were available. The donor count applies to the candidate estimate; observed readings remain unchanged.

## Supplied script and limitations

The supplied script also performs interpolation, Box-Cox transformation, seasonal decomposition, and outlier treatment before weekly averaging. Its weekly averaging block appears to include the donor-count column `N` in the sum, adding 1 to averages with at least one donor. Our implementation explicitly averages only the twelve shifted observation columns. It also parses timestamps explicitly and uses current pandas operations.

Weekly averages may smooth peaks and be inaccurate around holidays, changing demand, or unusual observed values. Estimates with few donors are less reliable. No artificial-masking experiment was performed, so these results demonstrate coverage rather than measured imputation accuracy.

This is retrospective cleaning: estimates use observations before and after a gap. For a forecasting task, split chronologically before designing imputation and avoid using future or test-set observations. Imputed target values should not be treated as observed ground truth for evaluation.

## Reproduce and inspect

Run `python 2_data_cleaning.py` from the project environment. Outputs are saved under `output/cleaning/`:

- `raw_consumption.png`: raw series for all feeders.
- `imputation_example.png`: original and filled values for an example week.
- `missing_summary.csv`: missing and imputed counts per feeder.
- `consumption_audit.csv`: complete hourly grid, flags, and unresolved gaps.
- `consumption_cleaned.csv`: readings remaining after imputation and omission of unresolved gaps.
