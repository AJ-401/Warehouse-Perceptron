# FIELD INTELLIGENCE — Godrej Hackathon — Streamlit Wrappers

Each of your 6 HTML pages has a matching Streamlit script that embeds it
exactly (same fonts, same Tailwind styling, same JS) using
`streamlit.components.v1.html()`, with CSS that stretches the embed to fill
your real browser window — the same fix used for the SIH V.A.U.L.T. pages,
applied from the start this time.

| # | Streamlit script    | Embeds              | Page                                          |
|---|-----------------------|----------------------|------------------------------------------------|
| 1 | `1_overview.py`       | `overview.html`      | Overview — Warehouse Video Intelligence (landing) |
| 2 | `2_behaviour.py`      | `behaviour.html`     | Behaviour Intelligence                          |
| 3 | `3_incidents.py`      | `incidents.html`     | Incident Investigation & Near-Miss Replay       |
| 4 | `4_intelligence.py`   | `intelligence.html`  | Autonomous Incident Mitigation                  |
| 5 | `5_prevention.py`     | `prevention.html`    | Prevention & Action Center                      |
| 6 | `6_assistance.py`     | `assistance.html`    | Operational Assistant                           |

(The numbering/order above is just a suggestion based on the page titles —
rename the files freely, order has no effect on how each one runs.)

## 1. One-time setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Important:** keep every `.py` file in the same folder as its matching
`.html` file — each script loads its HTML by filename from its own folder.

## 2. Run ONE page at a time

```bash
streamlit run 1_overview.py
```
Check it in your browser (http://localhost:8501), press `Ctrl+C` in the
terminal to stop it, then run the next one:

```bash
streamlit run 2_behaviour.py
streamlit run 3_incidents.py
streamlit run 4_intelligence.py
streamlit run 5_prevention.py
streamlit run 6_assistance.py
```

To run several at once instead, give each a different port:
```bash
streamlit run 1_overview.py --server.port 8501
streamlit run 2_behaviour.py --server.port 8502
```

## 3. Why it fits your window exactly

All 6 of these pages use a `min-h-screen flex flex-col` layout (unlike the
fixed `h-screen` pages from your other project), meaning they're built to
be *at least* one screen tall and grow if content needs more room. Each
script's CSS forces the embedded iframe to `width: 100vw; height: 100vh;
position: fixed`, so:
- On a normal amount of content, it fills your window exactly, like opening
  the HTML file directly.
- If a page's content ever grows taller than one screen (e.g. incidents.html
  with a long list of near-misses), the iframe scrolls internally rather
  than leaving dead space or stretching awkwardly.

## 4. These are still visual mockups

Like the previous set, buttons/links in these HTML files aren't wired to
real navigation or backend logic yet. This Streamlit wrapper reproduces the
*look* exactly. When you're ready to make the "Investigate", "Assist", or
alert-acknowledgement actions actually do something in Python, let me know
and I'll wire up real interactivity (Streamlit-native forms/session state
alongside the embedded HTML, since an iframe is sandboxed from the page
around it by default).
