[README.md](https://github.com/user-attachments/files/29479172/README.md)
# data/templates/ — Input Templates

Empty templates showing the **raw measured values** the pipeline expects as input.
No real patient data is included — only column headers, one illustrative example
row, and a data dictionary.

> ⚠️ These are raw inputs only. Derived features (ratios such as
> `hgb_g_d_l_div_mcv_f_l`, log transforms, etc.) are computed inside the
> pipeline — do **not** add them to your input file.

## Files

| File | Scenario | Raw columns |
|------|----------|-------------|
| `input_template_CBC_only.*` | CBC-only classifiers | 14 CBC parameters |
| `input_template_CBC_biochemistry.*` | CBC + biochemistry classifiers | 14 CBC + 4 biochemistry |

Each is provided as both `.csv` and `.xlsx`. The `.xlsx` files include
cell comments (unit + description on each header) and a `data_dictionary` sheet.

## Columns

**CBC (both scenarios):**
`patient_id, yas, hgb_g_d_l, rbc_10_6_u_l, mcv_f_l, mchc_g_dl, rdw_sd_fl,
ret_he_pg, delta_he_pg, ret_number_10_6_l, irf_pct, frc_perc, nrbc_pct,
micro_macro_ratio`

**Biochemistry (CBC+biochemistry scenario only):**
`ferritin, demir, ldh, uibc`

**Optional label columns** (`TARGET`, `DIAGNOSIS`) — fill only for
training/evaluation; leave blank for inference.

## Notes

- `nrbc_pct` is used by the Stage 1 classifier only; keep the column present.
- `micro_macro_ratio` is a direct Sysmex analyzer output, supplied as-is.
- Units follow the local laboratory reporting convention (see the
  `data_dictionary` sheet). Convert your data to matching units before running.
- The example row contains illustrative values only and must be deleted before
  using the template with your own data.
