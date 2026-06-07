# Interactive Web Report

A single-page HTML report of the entire CS228 project, ready to demo from a
laptop without any server. Built with UCR brand colors, embedded figures
(base64) and one interactive Plotly chart.

## How to use

**For presentation / demoing**:
```
Just double-click web/index.html in a file browser.
The whole report opens in your default browser — works offline.
```

**For sharing online**:
```bash
# Option 1 — upload to GitHub Pages (free)
cd CS228-Project
git checkout -b gh-pages
git add web/
git commit -m "add web report"
git push origin gh-pages
# Then enable Pages in repo Settings → Pages → branch gh-pages, folder /web

# Option 2 — preview locally on a port
cd web && python -m http.server 8000
# open http://localhost:8000 in browser
```

## Files

- `index.html` — the main report (≈30 KB, all CSS inline, 10 figures + 1 Plotly chart)
- `dataset.html` — **interactive floor-plan of the 28 sensors** (click any
  room → live stats + 7-day time series, all updated from real training
  data). Linked from the report header and Section 2.
- `figures.js` — base64-encoded copies of all 10 figures + the 24-h
  schedule data (≈2.1 MB) for the main report
- `sensors.js` — per-sensor statistics + 1-week time series for the
  dataset visualization (≈60 KB)

## Updating the figures

After re-running experiments and regenerating `results/figures/*.png`,
rebuild `figures.js` from the project root:

```bash
python -c "
import base64, json, os, numpy as np
src = 'CS228/results/figures'
out = {f: base64.b64encode(open(os.path.join(src, f), 'rb').read()).decode()
       for f in sorted(os.listdir(src)) if f.endswith('.png')}
sched = np.load('CS228/results/tcdpmixer_schedule.npy').tolist()
with open('web/figures.js', 'w') as fh:
    fh.write('window.FIGURES = ' + json.dumps(out) + ';\n')
    fh.write('window.SCHEDULE = ' + json.dumps(sched) + ';\n')
print('rebuilt web/figures.js')
"
```

## What's in the report

11 sections matching FINAL_REPORT_DRAFT.md exactly:
1. Introduction & Contributions
2. Dataset & Background
3. Related Work
4. Methods
5.1 – 5.11 Experiments (11 sub-sections, all 10 figures + 1 interactive)
6. Best configuration
7. Discussion & 3 take-aways
8. Limitations

The interactive Plotly chart in §5.11 lets the audience hover over any
hour to read the four gate values — useful during live Q&A.
