CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-main: #f8faf9;
    --bg-card: #ffffff;
    --text-primary: #0f2418;
    --text-secondary: #2d4a38;
    --text-muted: #5f7a6a;
    --border-color: #e2ebe6;
    --header-border: #d4e4da;
    --stat-bg: linear-gradient(145deg, #ffffff, #f3f8f5);
    --alert-critical-bg: #fffbfb;
    --alert-high-bg: #fffaf5;
    --placeholder-bg: linear-gradient(145deg, #ffffff, #f5faf7);
    --input-bg: #ffffff;
    --input-border: #d0ddd5;
    --accent: #2e7d4e;
    --accent-soft: #e8f5ee;
}

[data-theme="dark"] {
    --bg-main: #0c1a11;
    --bg-card: #12251a;
    --text-primary: #f0f7f2;
    --text-secondary: #a7c5b1;
    --text-muted: #6e937a;
    --border-color: #21432f;
    --header-border: #21432f;
    --stat-bg: linear-gradient(145deg, #12251a, #0c1a11);
    --alert-critical-bg: #12251a;
    --alert-high-bg: #12251a;
    --placeholder-bg: linear-gradient(145deg, #12251a, #0c1a11);
}

/* ── Reset & Base ── */
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background: var(--bg-main) !important; }
.main, [data-testid="stAppViewContainer"] {
    background: var(--bg-main) !important;
    color: var(--text-primary) !important;
}
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 1rem;
    max-width: 1100px;
    background: transparent !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar (light) ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid var(--border-color) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
    color: var(--text-secondary) !important;
    font-size: 0.85rem;
}

/* ── Animations ── */
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(18px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}
@keyframes shimmer {
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
@keyframes pulseGlow {
  0%, 100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.0); }
  50%      { box-shadow: 0 0 0 8px rgba(220,38,38,0.08); }
}
@keyframes scaleIn {
  from { opacity: 0; transform: scale(0.92); }
  to   { opacity: 1; transform: scale(1); }
}
@keyframes counterUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes borderPulse {
  0%, 100% { border-left-color: #dc2626; }
  50%      { border-left-color: #fca5a5; }
}
@keyframes ringSweep {
  from { stroke-dashoffset: 565; }
}

/* ── Alert Banner ── */
.alert-strip {
    padding: 16px 20px; border-radius: 6px; display: flex; align-items: center;
    gap: 14px; margin-bottom: 14px;
    border: 1px solid var(--border-color); border-left: 5px solid;
    animation: fadeSlideUp 0.4s ease both;
}
.alert-strip.critical {
    background: var(--alert-critical-bg); border-left-color: #dc2626;
    animation: fadeSlideUp 0.4s ease both, borderPulse 2.5s ease-in-out infinite;
}
.alert-strip.high {
    background: var(--alert-high-bg); border-left-color: #ea580c;
}
.alert-strip .alert-icon { font-size: 1.3rem; flex-shrink: 0; }
.alert-strip .alert-body h3 {
    margin: 0; font-size: 0.92rem; font-weight: 700; letter-spacing: 0.02em;
    color: var(--text-primary);
}
.alert-strip .alert-body p { margin: 3px 0 0 0; font-size: 0.84rem; font-weight: 400; color: var(--text-secondary); }

/* ── Header ── */
.header-bar {
    display: flex; align-items: center; gap: 14px;
    padding: 14px 0 12px 0; border-bottom: 2px solid var(--header-border); margin-bottom: 18px;
    animation: fadeIn 0.6s ease both;
}
.logo-mark {
    width: 40px; height: 40px; border-radius: 10px;
    background: linear-gradient(135deg, #124e2c 0%, #2e7d4e 100%);
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-weight: 800; font-size: 0.95rem;
    box-shadow: 0 2px 8px rgba(18,78,44,0.25);
    transition: transform 0.2s ease;
}
.logo-mark:hover { transform: rotate(-4deg) scale(1.05); }
.header-text {
    flex: 1;
}
.header-text h1 {
    margin: 0; font-size: 1.25rem; font-weight: 700; color: var(--text-primary);
    letter-spacing: -0.02em;
}
.header-text p { margin: 0; font-size: 0.76rem; color: var(--text-secondary); font-weight: 400; }
.header-lang-badge {
    background: var(--accent-soft);
    border: 1px solid var(--accent);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
}

/* Header columns styling */
div[data-testid="stHorizontalBlock"] > div > div {
    gap: 8px !important;
}

/* ── Cards ── */
.card {
    background: var(--bg-card); padding: 22px 24px; border-radius: 16px;
    border: 1px solid var(--border-color); margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(15, 36, 24, 0.04), 0 4px 16px rgba(15, 36, 24, 0.03);
    animation: fadeSlideUp 0.5s cubic-bezier(0.16,1,0.3,1) 0.1s both;
}
.card-title {
    font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
}
.card-title::before {
    content: ''; width: 3px; height: 14px; border-radius: 2px;
    background: linear-gradient(180deg, #124e2c, #34a853);
}
.section-title {
    font-size: 0.8rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--accent); margin: 18px 0 12px 0;
    display: flex; align-items: center; gap: 8px;
    border-bottom: 1px solid var(--border-color); padding-bottom: 6px;
}

/* ── Risk Badge ── */
.risk-badge {
    padding: 24px; border-radius: 14px; text-align: center; border: 2px solid;
    animation: scaleIn 0.5s cubic-bezier(0.16,1,0.3,1) 0.15s both;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    position: relative; overflow: hidden;
}
.risk-badge::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.4) 0%, rgba(255,255,255,0) 100%);
    pointer-events: none;
}
.risk-badge:hover { transform: scale(1.01); }
.risk-badge .risk-level {
    font-size: 1.9rem; font-weight: 800; margin: 0; letter-spacing: -0.03em;
    position: relative; z-index: 1;
}
.risk-badge .risk-urdu {
    font-size: 1.25rem; font-weight: 600; margin: 6px 0 0 0;
    direction: rtl; opacity: 0.85; position: relative; z-index: 1;
}
.risk-badge.risk-low {
    background: linear-gradient(145deg, #f0fdf4, #dcfce7); border-color: #86efac; color: #166534;
    box-shadow: 0 4px 14px rgba(22,163,74,0.08);
}
[data-theme="dark"] .risk-badge.risk-low { background: linear-gradient(145deg, #064e3b, #065f46); color: #ecfdf5; border-color: #059669; }

.risk-badge.risk-medium {
    background: linear-gradient(145deg, #fefce8, #fef9c3); border-color: #fde047; color: #854d0e;
    box-shadow: 0 4px 14px rgba(202,138,4,0.08);
}
[data-theme="dark"] .risk-badge.risk-medium { background: linear-gradient(145deg, #713f12, #854d0e); color: #fefce8; border-color: #a16207; }

.risk-badge.risk-high {
    background: linear-gradient(145deg, #fff7ed, #fed7aa); border-color: #fb923c; color: #9a3412;
    box-shadow: 0 4px 14px rgba(234,88,12,0.1);
}
[data-theme="dark"] .risk-badge.risk-high { background: linear-gradient(145deg, #7c2d12, #9a3412); color: #fff7ed; border-color: #c2410c; }

.risk-badge.risk-critical {
    background: linear-gradient(145deg, #fef2f2, #fecaca); border-color: #f87171; color: #991b1b;
    box-shadow: 0 4px 14px rgba(220,38,38,0.12);
}
[data-theme="dark"] .risk-badge.risk-critical { background: linear-gradient(145deg, #7f1d1d, #991b1b); color: #fef2f2; border-color: #b91c1c; }

/* ── SVG Ring Gauge ── */
.ring-gauge {
    display: flex; flex-direction: column; align-items: center;
    padding: 14px 0 6px 0;
    animation: fadeIn 0.5s ease 0.25s both;
}
.ring-gauge svg { }
.ring-gauge .ring-track {
    fill: none; stroke: var(--border-color); stroke-width: 10; stroke-linecap: round;
}
.ring-gauge .ring-fill {
    fill: none; stroke-width: 10; stroke-linecap: round;
    transform: rotate(-90deg); transform-origin: center;
    animation: ringSweep 1.4s cubic-bezier(0.16,1,0.3,1) 0.3s both;
}
.ring-gauge .ring-ticks { fill: none; stroke: var(--text-muted); stroke-width: 1.5; }
.ring-gauge .ring-center-val {
    font-size: 2rem; font-weight: 800; letter-spacing: -0.03em;
    dominant-baseline: central; text-anchor: middle;
}
.ring-gauge .ring-center-lbl {
    font-size: 0.55rem; font-weight: 500; fill: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.1em;
    dominant-baseline: central; text-anchor: middle;
}
.ring-gauge .tick-label {
    font-size: 0.5rem; fill: var(--text-muted); font-weight: 500;
    dominant-baseline: central; text-anchor: middle;
}

/* ── Stat Row ── */
.stat-row {
    display: flex; gap: 10px; margin-top: 14px;
    animation: fadeSlideUp 0.5s cubic-bezier(0.16,1,0.3,1) 0.5s both;
}
.stat-item {
    flex: 1; background: var(--stat-bg);
    border: 1px solid var(--border-color); border-radius: 12px; padding: 14px 10px; text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.stat-item:hover {
    border-color: var(--text-muted);
}
.stat-item .stat-val {
    font-size: 1.2rem; font-weight: 700; color: var(--text-primary); margin: 0;
    animation: counterUp 0.4s ease 0.7s both;
}
.stat-item .stat-lbl {
    font-size: 0.65rem; color: var(--text-muted); font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.08em; margin: 4px 0 0 0;
}

/* ── Action Banner ── */
.action-banner {
    padding: 16px 20px; border-radius: 12px; margin-top: 14px;
    display: flex; align-items: flex-start; gap: 12px;
    animation: fadeSlideUp 0.5s cubic-bezier(0.16,1,0.3,1) 0.6s both;
    transition: transform 0.2s ease;
    border: 1px solid transparent;
}

.action-banner .action-dot {
    width: 8px; height: 8px; border-radius: 50%; margin-top: 6px; flex-shrink: 0;
    box-shadow: 0 0 0 3px rgba(0,0,0,0.06);
}
.action-banner .action-text {
    font-size: 0.88rem; font-weight: 450; color: var(--text-primary); margin: 0; line-height: 1.55;
}

/* ── Sensor Note ── */
.sensor-note {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border: 1px solid #bfdbfe; border-radius: 10px;
    padding: 12px 16px; font-size: 0.8rem; color: #1e40af;
    margin-top: 10px; line-height: 1.5;
    animation: fadeIn 0.4s ease both;
}
[data-theme="dark"] .sensor-note { background: linear-gradient(135deg, #1e3a8a, #1e40af); color: #dbeafe; border-color: #3b82f6; }

/* ── Placeholder ── */
.placeholder-card {
    background: var(--placeholder-bg);
    border: 2px dashed #b8cfc0; border-radius: 16px;
    padding: 60px 30px; text-align: center;
    box-shadow: 0 1px 3px rgba(15, 36, 24, 0.03);
    animation: fadeIn 0.6s ease both;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.placeholder-card:hover {
    border-color: var(--accent);
    box-shadow: 0 4px 20px rgba(46, 125, 78, 0.06);
}
.placeholder-card h3 {
    color: var(--text-secondary); font-weight: 600; margin: 0 0 8px 0; font-size: 1.05rem;
}
.placeholder-card p { color: var(--text-muted); margin: 0; font-size: 0.85rem; line-height: 1.6; }

/* ── Validation ── */
.validation-msg {
    background: #fffbeb; border-left: 4px solid #f59e0b;
    border-radius: 4px; padding: 14px 18px;
    font-size: 0.85rem; color: #78350f; font-weight: 500;
    animation: fadeIn 0.3s ease both;
}
[data-theme="dark"] .validation-msg { background: #451a03; color: #fef3c7; border-left-color: #d97706; }

/* ── Details Row ── */
.details-row {
    margin-top: 14px; padding: 12px 16px; background: var(--stat-bg);
    border: 1px solid var(--border-color); border-radius: 10px;
    font-size: 0.76rem; color: var(--text-secondary); line-height: 1.7;
    animation: fadeIn 0.5s ease 0.7s both;
}

/* ── Footer ── */
.app-footer {
    text-align: center; color: var(--text-muted); font-size: 0.7rem;
    padding: 14px 0 4px 0; border-top: 1px solid var(--border-color); margin-top: 20px;
    animation: fadeIn 0.5s ease 0.8s both;
}

/* ── Streamlit Overrides ── */
.stForm { border: none !important; padding: 0 !important; background: transparent !important; }

.stSelectbox label, .stNumberInput label, .stDateInput label, .stExpander label {
    font-size: 0.8rem !important; font-weight: 500 !important; color: var(--text-secondary) !important;
}

/* Inputs — force light appearance */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-testid="stNumberInput"] input,
div[data-testid="stDateInput"] input {
    background-color: var(--input-bg) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(46, 125, 78, 0.12) !important;
}
div[data-baseweb="popover"] {
    background: #ffffff !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 24px rgba(15, 36, 24, 0.08) !important;
}
div[data-baseweb="popover"] li {
    color: var(--text-primary) !important;
    background: #ffffff !important;
}
div[data-baseweb="popover"] li:hover {
    background: var(--accent-soft) !important;
}

[data-testid="stExpander"] {
    background: #fafcfa !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}

.stToggle label span {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
}

.stFormSubmitButton > button {
    width: 100%; border-radius: 12px !important; height: 3rem;
    background: linear-gradient(135deg, #124e2c 0%, #2e7d4e 100%) !important;
    color: #fff !important; font-weight: 600 !important; font-size: 0.92rem !important;
    border: none !important; letter-spacing: 0.02em;
    transition: all 0.25s cubic-bezier(0.16,1,0.3,1);
    box-shadow: 0 2px 8px rgba(18,78,44,0.2);
    position: relative; overflow: hidden;
}
.stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #0a351d 0%, #124e2c 100%) !important;
    box-shadow: 0 6px 20px rgba(18,78,44,0.3);
    transform: translateY(-1px);
}
.stFormSubmitButton > button:active {
    transform: translateY(0) scale(0.985);
    box-shadow: 0 2px 6px rgba(18,78,44,0.2);
}

div[data-testid="stVerticalBlockBorderWrapper"] { border: none !important; }

div[data-baseweb="select"], div[data-baseweb="select"] * { cursor: pointer !important; }
div[data-baseweb="select"] input { cursor: pointer !important; caret-color: transparent !important; }

/* Language RTL Support */
[data-lang="urdu"] .header-text, [data-lang="urdu"] .section-title, [data-lang="urdu"] .card-title, [data-lang="urdu"] .action-text, [data-lang="urdu"] .stat-lbl {
    direction: rtl; text-align: right;
}
[data-lang="urdu"] .header-bar { flex-direction: row-reverse; }
[data-lang="urdu"] .logo-mark { margin-left: 14px; margin-right: 0; }

/* Language Button Styling */
.stButton > button {
    background: linear-gradient(135deg, #124e2c 0%, #2e7d4e 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    padding: 6px 12px !important;
    transition: all 0.2s ease !important;
    height: 36px !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(18,78,44,0.2) !important;
    background: linear-gradient(135deg, #0a351d 0%, #124e2c 100%) !important;
}

/* Language Indicator */
.lang-indicator {
    background: var(--accent-soft);
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-primary);
    text-align: center;
    margin-top: 8px;
}

</style>
"""
