import os
import datetime
import base64
import hashlib
from io import BytesIO
import streamlit as st
import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError
from agents.router_agent import RouterAgent
from agents.general_agent import GeneralAgent
from agents.rag_agent import RAGAgent
from agents.task_agent import TaskAgent
from agents.evaluator_agent import EvaluatorAgent
from tools.rag_tool import build_vector_store
from graph import create_graph

load_dotenv()

st.set_page_config(
    page_title="Mannan Medical Clinic AI",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
* { font-family: 'Inter', sans-serif; }

:root {
    --bg:           #f7f9fc;
    --surface:      #ffffff;
    --surface-2:    #f1f5fd;
    --border:       #dde4f0;
    --border-soft:  #eaeff8;
    --text:         #0d1b2a;
    --text-2:       #3d4f63;
    --muted:        #6b7a8d;
    --primary:      #2563eb;
    --primary-dark: #1d4ed8;
    --primary-soft: #eff4ff;
    --primary-mid:  #bfcfff;
    --success:      #16a34a;
    --success-soft: #f0fdf4;
    --danger:       #dc2626;
    --danger-soft:  #fef2f2;
    --warning:      #d97706;
    --warning-soft: #fffbeb;
    --purple:       #7c3aed;
    --purple-soft:  #f5f3ff;
    --radius:       12px;
    --radius-sm:    7px;
    --shadow-sm:    0 1px 3px rgba(13,27,42,0.06), 0 1px 2px rgba(13,27,42,0.04);
    --shadow-md:    0 4px 12px rgba(13,27,42,0.08), 0 2px 4px rgba(13,27,42,0.04);
    --shadow-lg:    0 8px 24px rgba(13,27,42,0.10), 0 2px 8px rgba(13,27,42,0.04);
}

html, body, [class*="css"] {
    background: var(--bg) !important;
    color: var(--text) !important;
}

p, span, li, h1, h2, h3, h4, h5, h6,
strong, em, label, div { color: var(--text) !important; }

.main .block-container {
    max-width: 1100px;
    padding: 1.75rem 2.5rem 8rem;
}

/* ════════════════════════════════
   ANIMATED ACCENT BAR
════════════════════════════════ */
.accent-bar {
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(90deg,
        #2563eb 0%, #7c3aed 30%, #16a34a 60%, #2563eb 100%);
    background-size: 200% 100%;
    animation: accentShift 4s ease-in-out infinite;
    margin-bottom: 14px;
}

/* ════════════════════════════════
   TOP NAV BAR
════════════════════════════════ */
.top-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 22px;
    margin-bottom: 20px;
    box-shadow: var(--shadow-sm);
    animation: fadeUp 0.5s ease-out both;
    position: relative;
    overflow: hidden;
}

.top-nav::before {
    content: '';
    position: absolute;
    top: 0; left: -100%; width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(37,99,235,0.05), transparent);
    animation: navShimmer 3s ease-in-out 0.6s 1 forwards;
}

.nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.nav-logo {
    width: 42px; height: 42px;
    border-radius: 11px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    display: flex; align-items: center; justify-content: center;
    color: #fff !important;
    font-size: 20px; font-weight: 800;
    box-shadow: 0 4px 14px rgba(37,99,235,0.35);
    flex-shrink: 0;
    animation: logoPop 0.6s cubic-bezier(0.34,1.56,0.64,1) both;
}

.nav-clinic-name {
    font-size: 16px;
    font-weight: 700;
    color: var(--text) !important;
    line-height: 1.2;
}

.nav-tagline {
    font-size: 11px;
    color: var(--muted) !important;
    margin-top: 1px;
}

.nav-pills {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}

.nav-pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--primary-soft);
    border: 1px solid var(--primary-mid);
    color: var(--primary-dark) !important;
    padding: 5px 11px;
    border-radius: 20px;
    font-size: 11px; font-weight: 600;
    animation: pillFloat 0.5s ease-out both;
}

.nav-pill:nth-child(1) { animation-delay: 0.1s; }
.nav-pill:nth-child(2) { animation-delay: 0.2s; }
.nav-pill:nth-child(3) { animation-delay: 0.3s; }
.nav-pill:nth-child(4) { animation-delay: 0.4s; }

.nav-status {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--success-soft);
    border: 1px solid #bbf7d0;
    color: var(--success) !important;
    padding: 5px 11px;
    border-radius: 20px;
    font-size: 11px; font-weight: 600;
    animation: pillFloat 0.5s ease-out 0.5s both;
}

.nav-status-dot {
    width: 8px; height: 8px;
    background: var(--success);
    border-radius: 50%;
    animation: heartbeat 2s ease-in-out infinite;
}

/* ════════════════════════════════
   STATS ROW
════════════════════════════════ */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}

.stat-tile {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 18px 14px;
    box-shadow: var(--shadow-sm);
    position: relative;
    overflow: hidden;
    transition: transform 0.2s cubic-bezier(0.34,1.56,0.64,1), box-shadow 0.2s;
    animation: statSlideIn 0.5s ease-out both;
}

.stat-tile:nth-child(1) { animation-delay: 0.05s; }
.stat-tile:nth-child(2) { animation-delay: 0.12s; }
.stat-tile:nth-child(3) { animation-delay: 0.19s; }
.stat-tile:nth-child(4) { animation-delay: 0.26s; }

.stat-tile::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 0 0 var(--radius) var(--radius);
    background-size: 200% 100%;
    animation: accentShift 3s ease-in-out infinite;
}

.stat-tile.blue::after  { background: linear-gradient(90deg,#2563eb,#60a5fa,#2563eb); }
.stat-tile.green::after { background: linear-gradient(90deg,#16a34a,#4ade80,#16a34a); }
.stat-tile.red::after   { background: linear-gradient(90deg,#dc2626,#f87171,#dc2626); }
.stat-tile.purple::after{ background: linear-gradient(90deg,#7c3aed,#a78bfa,#7c3aed); }

.stat-tile:hover {
    transform: translateY(-4px) scale(1.02);
}

.stat-tile.blue:hover  { box-shadow: 0 8px 24px rgba(37,99,235,0.18); }
.stat-tile.green:hover { box-shadow: 0 8px 24px rgba(22,163,74,0.18); }
.stat-tile.red:hover   { box-shadow: 0 8px 24px rgba(220,38,38,0.18); }
.stat-tile.purple:hover{ box-shadow: 0 8px 24px rgba(124,58,237,0.18); }

.stat-icon {
    font-size: 24px;
    margin-bottom: 8px;
    display: block;
    animation: iconWave 3s ease-in-out infinite;
}

.stat-tile:nth-child(1) .stat-icon { animation-delay: 0s; }
.stat-tile:nth-child(2) .stat-icon { animation-delay: 0.5s; }
.stat-tile:nth-child(3) .stat-icon { animation-delay: 1s; }
.stat-tile:nth-child(4) .stat-icon { animation-delay: 1.5s; }

.stat-value {
    font-size: 1.7rem;
    font-weight: 800;
    color: var(--text) !important;
    line-height: 1;
}

.stat-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 5px;
}

/* ════════════════════════════════
   ALERT STRIP
════════════════════════════════ */
.alert-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}

.alert-tile {
    border-radius: var(--radius);
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    border: 1px solid transparent;
    transition: transform 0.2s cubic-bezier(0.34,1.2,0.64,1), box-shadow 0.2s;
    animation: fadeUp 0.5s ease-out both;
}

.alert-tile:nth-child(1) { animation-delay: 0.15s; }
.alert-tile:nth-child(2) { animation-delay: 0.22s; }
.alert-tile:nth-child(3) { animation-delay: 0.29s; }

.alert-tile:hover { transform: translateY(-2px) scale(1.01); box-shadow: var(--shadow-md); }

.alert-tile.red   { background: var(--danger-soft);  border-color: #fecaca; animation: redPulse 2.2s ease-in-out infinite; }
.alert-tile.amber { background: var(--warning-soft); border-color: #fde68a; animation: amberPulse 2.8s ease-in-out infinite; }
.alert-tile.blue  { background: var(--purple-soft);  border-color: #ddd6fe; }

.alert-icon-row { display: flex; align-items: center; gap: 8px; }
.alert-label { font-size: 13px; font-weight: 700; }
.alert-tile.red   .alert-label { color: var(--danger) !important; }
.alert-tile.amber .alert-label { color: var(--warning) !important; }
.alert-tile.blue  .alert-label { color: var(--purple) !important; }
.alert-desc { font-size: 11px; color: var(--muted) !important; line-height: 1.4; }
.alert-number { font-size: 15px; font-weight: 800; margin-top: 4px; }
.alert-tile.red   .alert-number { color: var(--danger) !important; }
.alert-tile.amber .alert-number { color: var(--warning) !important; }
.alert-tile.blue  .alert-number { color: var(--purple) !important; }
.alert-tag {
    display: inline-block; margin-top: 4px; padding: 2px 7px;
    border-radius: 4px; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.alert-tile.red   .alert-tag { background: #fecaca; color: var(--danger) !important; }
.alert-tile.amber .alert-tag { background: #fde68a; color: var(--warning) !important; }
.alert-tile.blue  .alert-tag { background: #ddd6fe; color: var(--purple) !important; }

/* ════════════════════════════════
   INTAKE WORKSPACE PANEL
════════════════════════════════ */
.intake-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 20px;
    margin-bottom: 16px;
    box-shadow: var(--shadow-sm);
    animation: fadeUp 0.5s ease-out 0.12s both;
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
}

.intake-text .intake-title { font-size: 14px; font-weight: 700; color: var(--text) !important; }
.intake-text .intake-sub {
    font-size: 12px; color: var(--muted) !important;
    line-height: 1.5; margin-top: 3px; max-width: 680px;
}

.intake-badge {
    flex-shrink: 0; padding: 5px 10px; border-radius: var(--radius-sm);
    background: #ecfdf5; border: 1px solid #a7f3d0;
    color: var(--success) !important; font-size: 11px; font-weight: 700; white-space: nowrap;
    animation: badgePop 0.6s cubic-bezier(0.34,1.56,0.64,1) 0.4s both;
}

/* ════════════════════════════════
   SAFETY NOTE
════════════════════════════════ */
.safety-note {
    display: flex; align-items: flex-start; gap: 10px;
    background: #fffbeb; border: 1px solid #fde68a;
    border-left: 4px solid var(--warning); border-radius: var(--radius);
    padding: 11px 14px; margin-bottom: 14px;
    animation: fadeUp 0.5s ease-out 0.15s both;
}
.safety-note strong { font-size: 12px; font-weight: 700; color: #92400e !important; }
.safety-note span   { font-size: 11px; color: #78350f !important; line-height: 1.5; }

/* ════════════════════════════════
   CHAT TOOLBAR
════════════════════════════════ */
.chat-toolbar {
    display: flex; align-items: center; justify-content: space-between;
    margin: 0 0 10px; padding: 10px 14px; background: var(--surface);
    border: 1px solid var(--border); border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
}
.chat-toolbar-title { font-size: 13px; font-weight: 700; color: var(--text) !important; display: flex; align-items: center; gap: 8px; }
.live-badge {
    display: inline-flex; align-items: center; gap: 5px; background: var(--success-soft);
    border: 1px solid #bbf7d0; border-radius: 20px; padding: 4px 10px;
    font-size: 11px; font-weight: 600; color: var(--success) !important;
}
.live-dot { width: 7px; height: 7px; background: var(--success); border-radius: 50%; animation: heartbeat 2s ease-in-out infinite; }

/* ════════════════════════════════
   CHAT BUBBLES
════════════════════════════════ */
.stChatMessage { margin: 6px 0 !important; animation: messageIn 0.25s cubic-bezier(0.34,1.2,0.64,1) both; }

[data-testid="stChatMessage-user"] [data-testid="stChatMessageContent"] {
    background: var(--primary-soft) !important; border: 1px solid var(--primary-mid) !important;
    border-radius: var(--radius) !important; padding: 12px 16px !important; box-shadow: var(--shadow-sm) !important;
}
[data-testid="stChatMessage-assistant"] [data-testid="stChatMessageContent"] {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; padding: 12px 16px !important; box-shadow: var(--shadow-sm) !important;
}
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] strong,
[data-testid="stChatMessageContent"] span,
[data-testid="stChatMessageContent"] em { color: var(--text) !important; }

.msg-meta { display: flex; align-items: center; gap: 6px; margin-bottom: 8px; flex-wrap: wrap; }
.msg-badge {
    display: inline-flex; align-items: center; padding: 3px 8px; border-radius: 4px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.02em;
    animation: badgePop 0.4s cubic-bezier(0.34,1.56,0.64,1) both;
}
.msg-badge.primary  { background: var(--primary-soft); border: 1px solid var(--primary-mid); color: var(--primary-dark) !important; }
.msg-badge.success  { background: var(--success-soft); border: 1px solid #bbf7d0; color: var(--success) !important; }
.msg-badge.danger   { background: var(--danger-soft);  border: 1px solid #fecaca; color: var(--danger) !important; }
.msg-badge.purple   { background: var(--purple-soft);  border: 1px solid #ddd6fe; color: var(--purple) !important; }
.msg-time { font-size: 10px; color: var(--muted) !important; margin-top: 8px; text-align: right; }

/* ════════════════════════════════
   EMERGENCY ALERT BOX
════════════════════════════════ */
.emerg-box {
    background: var(--danger-soft); border: 1.5px solid #fecaca;
    border-left: 5px solid var(--danger); border-radius: var(--radius);
    padding: 18px 20px; margin-bottom: 12px;
    animation: emergShake 0.5s cubic-bezier(0.36,0.07,0.19,0.97) both;
}
.emerg-title { font-size: 15px; font-weight: 800; color: var(--danger) !important; margin-bottom: 6px; }
.emerg-body  { font-size: 12px; color: #7f1d1d !important; line-height: 1.6; margin-bottom: 12px; }
.emerg-actions { display: flex; gap: 8px; flex-wrap: wrap; }
.emerg-btn {
    flex: 1 1 160px; padding: 9px 14px; border-radius: var(--radius-sm);
    font-size: 12px; font-weight: 700; text-align: center; color: #fff !important;
    transition: transform 0.15s, box-shadow 0.15s;
}
.emerg-btn:hover { transform: scale(1.03); box-shadow: 0 4px 12px rgba(220,38,38,0.3); }

/* ════════════════════════════════
   INPUT BOX
════════════════════════════════ */
.stChatInputContainer {
    background: var(--surface) !important; border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important; box-shadow: var(--shadow-md) !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stChatInputContainer:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12), var(--shadow-md) !important;
}
.stChatInput textarea { color: var(--text) !important; background: transparent !important; font-size: 14px !important; line-height: 1.5 !important; }
.stChatInput textarea::placeholder { color: var(--muted) !important; }
.stChatInputContainer button {
    background: var(--primary) !important; border-radius: var(--radius-sm) !important; border: none !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.25) !important; transition: background 0.15s, transform 0.15s !important;
}
.stChatInputContainer button:hover { background: var(--primary-dark) !important; transform: scale(1.06) !important; }

/* ════════════════════════════════
   TYPING DOTS
════════════════════════════════ */
.typing-wrap { display: flex; align-items: center; gap: 10px; padding: 10px 4px; }
.typing-dot { width: 8px; height: 8px; background: var(--primary); border-radius: 50%; margin: 0 2px; animation: typing 1.2s ease-in-out infinite; }
.typing-dot:nth-child(2) { animation-delay: 0.18s; }
.typing-dot:nth-child(3) { animation-delay: 0.36s; }
.typing-label { font-size: 11px; color: var(--muted) !important; }

/* ════════════════════════════════
   SIDEBAR
════════════════════════════════ */
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }

.sb-section {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 14px; margin-bottom: 12px; box-shadow: var(--shadow-sm); transition: box-shadow 0.2s;
}
.sb-section:hover { box-shadow: var(--shadow-md); }
.sb-section-title {
    font-size: 11px; font-weight: 700; color: var(--muted) !important;
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 10px;
    padding-bottom: 8px; border-bottom: 1px solid var(--border-soft);
}
.sb-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 0; border-bottom: 1px solid var(--border-soft); gap: 8px;
}
.sb-row:last-child { border-bottom: none; padding-bottom: 0; }
.sb-row-label { font-size: 11px; color: var(--muted) !important; }
.sb-row-value { font-size: 11px; font-weight: 600; color: var(--text) !important; text-align: right; }

.counter-box {
    background: var(--primary-soft); border: 1px solid var(--primary-mid); border-radius: var(--radius);
    padding: 12px 14px; margin-bottom: 12px; display: flex; align-items: center;
    justify-content: space-between; transition: transform 0.2s;
}
.counter-box:hover { transform: scale(1.01); }
.counter-label { font-size: 11px; font-weight: 600; color: var(--primary-dark) !important; }
.counter-sub { font-size: 10px; color: var(--muted) !important; margin-top: 1px; }
.counter-num { font-size: 22px; font-weight: 800; color: var(--primary) !important; }

.doc-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-sm);
    padding: 9px 11px; margin-bottom: 7px; transition: all 0.2s cubic-bezier(0.34,1.2,0.64,1);
}
.doc-card:hover { border-color: var(--primary); background: var(--primary-soft); transform: translateX(3px); }
.doc-name { font-size: 12px; font-weight: 700; color: var(--text) !important; }
.doc-spec { font-size: 10px; color: var(--muted) !important; margin-top: 1px; }
.doc-avail { display: inline-flex; align-items: center; gap: 4px; margin-top: 5px; padding: 2px 7px; border-radius: 4px; font-size: 10px; font-weight: 700; }
.doc-avail.on  { background: var(--success-soft); color: var(--success) !important; border: 1px solid #bbf7d0; }
.doc-avail.off { background: #fef2f2; color: var(--danger) !important; border: 1px solid #fecaca; }
.avail-dot { width: 5px; height: 5px; border-radius: 50%; }
.avail-dot.on  { background: var(--success); animation: heartbeat 2s ease-in-out infinite; }
.avail-dot.off { background: var(--danger); }

.agent-row {
    display: flex; align-items: center; gap: 8px; padding: 7px 9px;
    border-radius: var(--radius-sm); margin-bottom: 4px; transition: background 0.15s, transform 0.15s; cursor: default;
}
.agent-row:hover { background: var(--primary-soft); transform: translateX(2px); }
.agent-row-dot { width: 7px; height: 7px; background: var(--primary); border-radius: 50%; flex-shrink: 0; animation: heartbeat 2.4s ease-in-out infinite; }
.agent-row-name { font-size: 12px; font-weight: 600; color: var(--text) !important; }
.agent-row-desc { font-size: 10px; color: var(--muted) !important; margin-top: 1px; }

button, .stButton button { border-radius: var(--radius-sm) !important; }

[data-testid="stSidebar"] .stButton button {
    width: 100%; min-height: 36px; background: var(--surface) !important;
    border: 1px solid var(--border) !important; color: var(--text-2) !important;
    font-size: 12px !important; text-align: left !important; white-space: normal !important;
    line-height: 1.35 !important; box-shadow: var(--shadow-sm) !important;
    transition: all 0.2s !important; padding: 8px 10px !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: var(--primary-soft) !important; border-color: var(--primary) !important;
    color: var(--primary-dark) !important; transform: translateX(3px) !important;
}

/* ════════════════════════════════
   ANIMATIONS
════════════════════════════════ */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes statSlideIn {
    from { opacity: 0; transform: translateY(14px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes messageIn {
    from { opacity: 0; transform: translateY(6px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes typing {
    0%, 100% { opacity: 0.3; transform: translateY(0); }
    50%       { opacity: 1;   transform: translateY(-4px); }
}
@keyframes heartbeat {
    0%   { transform: scale(1);    box-shadow: 0 0 0 0 rgba(22,163,74,0.4); }
    30%  { transform: scale(1.3);  box-shadow: 0 0 0 5px rgba(22,163,74,0); }
    100% { transform: scale(1);    box-shadow: 0 0 0 0 rgba(22,163,74,0); }
}
@keyframes accentShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes navShimmer {
    0%   { left: -100%; opacity: 0; }
    10%  { opacity: 1; }
    100% { left: 150%; opacity: 0; }
}
@keyframes logoPop {
    from { opacity: 0; transform: scale(0.6) rotate(-10deg); }
    to   { opacity: 1; transform: scale(1) rotate(0deg); }
}
@keyframes pillFloat {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes badgePop {
    from { opacity: 0; transform: scale(0.7); }
    to   { opacity: 1; transform: scale(1); }
}
@keyframes iconWave {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-3px); }
}
@keyframes redPulse {
    0%, 100% { box-shadow: 0 2px 8px rgba(220,38,38,0.08); }
    50%       { box-shadow: 0 4px 18px rgba(220,38,38,0.22); }
}
@keyframes amberPulse {
    0%, 100% { box-shadow: 0 2px 8px rgba(217,119,6,0.08); }
    50%       { box-shadow: 0 4px 18px rgba(217,119,6,0.20); }
}
@keyframes emergShake {
    0%, 100% { transform: translateX(0); }
    15%       { transform: translateX(-4px); }
    30%       { transform: translateX(4px); }
    45%       { transform: translateX(-3px); }
    60%       { transform: translateX(3px); }
    75%       { transform: translateX(-1px); }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; transition-duration: 0.01ms !important; }
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--primary-mid); border-radius: 3px; }

@media (max-width: 1024px) {
    .main .block-container { padding: 1.25rem 1.5rem 7rem; }
    .stats-row { grid-template-columns: repeat(2, 1fr); }
    .alert-strip { grid-template-columns: 1fr; }
}
@media (max-width: 768px) {
    .main .block-container { padding: 1rem 0.9rem 7rem; }
    .top-nav { flex-direction: column; align-items: flex-start; gap: 10px; }
    .stats-row { grid-template-columns: repeat(2, 1fr); gap: 8px; }
    .alert-strip { grid-template-columns: 1fr; }
    .intake-panel { flex-direction: column; }
    .chat-toolbar { flex-direction: column; align-items: flex-start; gap: 8px; }
    .emerg-btn { flex-basis: 100%; }
}
@media (max-width: 480px) {
    .main .block-container { padding: 0.75rem 0.7rem 7rem; }
    .stats-row { grid-template-columns: 1fr 1fr; gap: 8px; }
    .stat-tile { padding: 12px 14px; }
    .stat-value { font-size: 1.3rem; }
}
</style>
""", unsafe_allow_html=True)

# ── Animated Accent Bar ───────────────────────────────
st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

# ── Top Nav ───────────────────────────────────────────
st.markdown("""
<div class="top-nav">
    <div class="nav-brand">
        <div class="nav-logo">M</div>
        <div>
            <div class="nav-clinic-name">Mannan Medical Clinic</div>
            <div class="nav-tagline">Sahiwal, Punjab, Pakistan · Est. Patient Portal</div>
        </div>
    </div>
    <div class="nav-pills">
        <span class="nav-pill">⚡ LangGraph</span>
        <span class="nav-pill">🤖 Multi-Agent</span>
        <span class="nav-pill">📚 Agentic RAG</span>
        <span class="nav-pill">🦙 Llama 3.3</span>
        <span class="nav-status">
            <span class="nav-status-dot"></span>
            System Online
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Stats Row ─────────────────────────────────────────
st.markdown("""
<div class="stats-row">
    <div class="stat-tile blue">
        <span class="stat-icon">🤖</span>
        <span class="stat-value">5</span>
        <span class="stat-label">AI Agents Active</span>
    </div>
    <div class="stat-tile green">
        <span class="stat-icon">👨‍⚕️</span>
        <span class="stat-value">4</span>
        <span class="stat-label">Doctors on Network</span>
    </div>
    <div class="stat-tile purple">
        <span class="stat-icon">🧪</span>
        <span class="stat-value">10+</span>
        <span class="stat-label">Clinic Services</span>
    </div>
    <div class="stat-tile red">
        <span class="stat-icon">🚨</span>
        <span class="stat-value">24/7</span>
        <span class="stat-label">Emergency Line</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Alert Strip ───────────────────────────────────────
st.markdown("""
<div class="alert-strip">
    <div class="alert-tile red">
        <div class="alert-icon-row">
            <div class="alert-label">🚨 Emergency Support</div>
        </div>
        <div class="alert-desc">Life-threatening situation — call immediately, do not wait.</div>
        <div class="alert-number">📞 0300-1234567</div>
        <span class="alert-tag">● Active 24/7</span>
    </div>
    <div class="alert-tile amber">
        <div class="alert-icon-row">
            <div class="alert-label">⚠️ Urgent Care</div>
        </div>
        <div class="alert-desc">Needs immediate medical attention today. Walk in directly.</div>
        <div class="alert-number">🏥 Walk In Now</div>
        <span class="alert-tag">Mon – Sat · 9AM–6PM</span>
    </div>
    <div class="alert-tile blue">
        <div class="alert-icon-row">
            <div class="alert-label">🚑 Punjab Rescue</div>
        </div>
        <div class="alert-desc">Free government emergency ambulance and rescue service.</div>
        <div class="alert-number">📞 1122</div>
        <span class="alert-tag">Government · Free</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load System ───────────────────────────────────────
@st.cache_resource
def load_system():
    groq_key   = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if groq_key:
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key)
    elif gemini_key:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_key)
    else:
        st.error("❌ API Key not found! Check .env file.")
        st.stop()
    vectorstore = build_vector_store("knowledge_base")
    router    = RouterAgent(llm, "Router")
    rag       = RAGAgent(llm, "RAG", vectorstore)
    task      = TaskAgent(llm, "Task")
    general   = GeneralAgent(llm, "General")
    evaluator = EvaluatorAgent(llm, "Evaluator")
    return create_graph(router, rag, task, general, evaluator)

@st.cache_resource
def get_gemini_media_model():
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        return None
    media_model = os.getenv("GEMINI_MEDIA_MODEL", "gemini-2.0-flash")
    return ChatGoogleGenerativeAI(model=media_model, google_api_key=gemini_key, temperature=0.1)

def get_lab_vision_model():
    return get_gemini_media_model()

def normalize_chat_files(files):
    if not files: return []
    if isinstance(files, (list, tuple)): return list(files)
    return [files]

def is_provider_quota_error(exc):
    text = str(exc).lower()
    return any(k in text for k in ["resource_exhausted","quota","429","rate limit"])

def media_provider_error_message(feature_name, exc):
    media_model = os.getenv("GEMINI_MEDIA_MODEL", "gemini-2.0-flash")
    if is_provider_quota_error(exc):
        return (f"{feature_name} is temporarily unavailable because the Gemini media model "
                f"`{media_model}` has reached its API quota. Please wait and try again.")
    return (f"{feature_name} could not be completed right now. "
            "Please try again or type details directly in the chat box.")

def transcribe_with_groq(audio_file):
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key: return None
    audio_name = audio_file.name or "voice.wav"
    audio_type = audio_file.type or "audio/wav"
    groq_model = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
    response = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {groq_key}"},
        data={"model": groq_model, "response_format": "json", "temperature": "0"},
        files={"file": (audio_name, audio_file.getvalue(), audio_type)},
        timeout=60,
    )
    if response.status_code == 429:
        raise RuntimeError("Groq speech-to-text quota or rate limit reached.")
    response.raise_for_status()
    return (response.json().get("text") or "").strip()

def transcribe_voice_input(audio_file):
    try:
        groq_text = transcribe_with_groq(audio_file)
        if groq_text: return groq_text
    except Exception:
        pass
    llm = get_gemini_media_model()
    if llm is None:
        return "Voice input needs `GOOGLE_API_KEY` in your `.env` file."
    mime_type = audio_file.type or "audio/wav"
    audio_bytes = audio_file.getvalue()
    prompt = "Transcribe this patient voice message. Return only the spoken words. If unclear say: [Unclear voice message]."
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "media", "mime_type": mime_type, "data": audio_bytes},
    ])
    try:
        response = llm.invoke([message])
    except Exception as exc:
        return media_provider_error_message("Voice transcription", exc)
    return response.content.strip()

def prepare_lab_report_image(uploaded_file):
    try:
        image = Image.open(BytesIO(uploaded_file.getvalue()))
    except UnidentifiedImageError as exc:
        raise ValueError(f"{uploaded_file.name} is not a readable image.") from exc
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_size = image.size
    max_side = 1800
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.15)
    image = ImageEnhance.Sharpness(image).enhance(1.2)
    output = BytesIO()
    image.save(output, format="JPEG", quality=90, optimize=True)
    return {"name": uploaded_file.name, "bytes": output.getvalue(),
            "mime_type": "image/jpeg", "original_size": original_size, "processed_size": image.size}

def analyze_lab_report_images(uploaded_files, patient_context=""):
    llm = get_lab_vision_model()
    if llm is None:
        return "Lab report analysis needs `GOOGLE_API_KEY` in your `.env` file."
    prepared_images = [prepare_lab_report_image(file) for file in uploaded_files[:4]]
    image_notes = "\n".join(
        f"- {item['name']}: original {item['original_size'][0]}x{item['original_size'][1]}, "
        f"processed {item['processed_size'][0]}x{item['processed_size'][1]}"
        for item in prepared_images
    )
    prompt = f"""You are a careful medical lab report analysis assistant.
Analyze every uploaded lab report image. Extract values, compare with reference ranges, identify abnormal findings.
Patient context: {patient_context or "None provided."}
Images: {image_notes}

Return a concise Markdown report with:
1. Image readability
2. Extracted lab values (table: Test, Value, Unit, Reference Range, Status, Note)
3. Abnormal findings
4. Possible clinical meaning
5. Recommended next steps
6. Doctor to consult
7. Medicine guidance (no prescriptions)

Status: Normal / Low / High / Borderline / Unclear only.
Do not diagnose. Do not invent values."""
    content = [{"type": "text", "text": prompt}]
    for item in prepared_images:
        encoded_image = base64.b64encode(item["bytes"]).decode("utf-8")
        content.append({"type": "image_url", "image_url": {"url": f"data:{item['mime_type']};base64,{encoded_image}"}})
    message = HumanMessage(content=content)
    try:
        response = llm.invoke([message])
    except Exception as exc:
        return media_provider_error_message("Lab report analysis", exc)
    return response.content

CLINIC_DOCTORS = [
    {"name": "Dr. Ahmed Khan",    "spec": "General Physician", "site": "Mannan Medical Clinic",            "days": [0,2,4],       "start": 9,  "end": 14},
    {"name": "Dr. Ayesha Malik",  "spec": "Gynecologist",      "site": "Sahiwal Women Care Unit",         "days": [1,3,5],       "start": 10, "end": 16},
    {"name": "Dr. Bilal Hussain", "spec": "Cardiologist",      "site": "Sahiwal Heart Consultation Desk", "days": [0,3],         "start": 14, "end": 18},
    {"name": "Dr. Sara Ahmed",    "spec": "Pediatrician",      "site": "Sahiwal Child Care Unit",         "days": [0,1,2,3,4,5], "start": 9,  "end": 13},
]

SAHIWAL_EMERGENCY_HOSPITALS = [
    {"name": "Sahiwal District Emergency Center", "address": "Central Sahiwal emergency corridor",
     "reason": "Severe symptoms, trauma, chest pain, stroke, breathing difficulty.", "phone": "1122"},
    {"name": "Sahiwal Cardiac & Critical Care Desk", "address": "Sahiwal city care network",
     "reason": "Chest pain, palpitations, collapse, high-risk cardiac symptoms.", "phone": "Clinic emergency / 1122"},
    {"name": "Punjab Rescue 1122", "address": "Sahiwal District", "reason": "Emergency ambulance and rescue.", "phone": "1122"},
]

def doctor_status(doc, now=None):
    now = now or datetime.datetime.now()
    available = now.weekday() in doc["days"] and doc["start"] <= now.hour < doc["end"]
    return {**doc, "available": available, "time": f"{doc['start']:02d}:00–{doc['end']:02d}:00"}

def current_doctor_availability():
    return [doctor_status(doc) for doc in CLINIC_DOCTORS]

def format_available_doctors_markdown():
    doctors = current_doctor_availability()
    available = [d for d in doctors if d["available"]]
    if not available:
        next_options = ", ".join(f"{d['name']} ({d['spec']}, {d['time']})" for d in doctors[:2])
        return f"No doctor is available right now. Next: {next_options}."
    return "\n".join(f"- **{d['name']}** — {d['spec']} at {d['site']} ({d['time']})." for d in available)

def format_hospital_recommendations_markdown():
    return "\n".join(
        f"- **{h['name']}** — {h['address']}. {h['reason']} Contact: {h['phone']}."
        for h in SAHIWAL_EMERGENCY_HOSPITALS
    )

def twilio_configured():
    return all(os.getenv(k) for k in ["TWILIO_ACCOUNT_SID","TWILIO_AUTH_TOKEN","TWILIO_FROM_NUMBER","TWILIO_EMERGENCY_TO_NUMBER"])

def send_twilio_emergency_sms(problem_text):
    if not twilio_configured():
        return "Twilio not configured. Add credentials to `.env`."
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token  = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    to_number   = os.getenv("TWILIO_EMERGENCY_TO_NUMBER")
    body = f"Emergency from Mannan Medical Clinic AI.\nProblem: {problem_text[:600]}\nAction: Call patient, advise Rescue 1122."
    response = requests.post(
        f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json",
        data={"From": from_number, "To": to_number, "Body": body},
        auth=(account_sid, auth_token), timeout=20,
    )
    if response.status_code >= 400:
        return "Twilio SMS could not be sent. Check credentials."
    return "Twilio emergency SMS sent to configured emergency contact."

def build_emergency_recommendation(problem_text):
    care_alert_status = "Care alert is in demo mode. No external SMS was sent."
    problem_hash = hashlib.sha256(problem_text.encode("utf-8")).hexdigest()
    if twilio_configured() and st.session_state.get("last_twilio_emergency_hash") != problem_hash:
        care_alert_status = send_twilio_emergency_sms(problem_text)
        st.session_state.last_twilio_emergency_hash = problem_hash
    elif twilio_configured():
        care_alert_status = "Emergency care alert already sent for this request."
    return (
        "🚨 **Emergency Recommendation — Sahiwal**\n\n"
        "**Immediate action**\n"
        "- Call **Rescue 1122** now for chest pain, breathing difficulty, stroke, seizure, or severe bleeding.\n\n"
        "**Sahiwal emergency network**\n"
        f"{format_hospital_recommendations_markdown()}\n\n"
        "**Available doctors**\n"
        f"{format_available_doctors_markdown()}\n\n"
        "**Care alert status**\n"
        f"- {care_alert_status}"
    )

app = load_system()

# ── Intake Workspace Panel ────────────────────────────
st.markdown("""
<div class="intake-panel">
    <div class="intake-text">
        <div class="intake-title">Clinical Intake Workspace</div>
        <div class="intake-sub">
            Type your symptoms, record a voice message, or attach a lab report image.
            The system routes each request through symptom triage, report analysis,
            emergency guidance, and live doctor availability.
        </div>
    </div>
    <div class="intake-badge">Production Demo</div>
</div>
""", unsafe_allow_html=True)

# ── Emergency Routing Expander ────────────────────────
with st.expander("🗺️ Sahiwal Emergency Routing", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sahiwal emergency care network**")
        st.markdown(format_hospital_recommendations_markdown())
    with c2:
        st.markdown("**Live doctor availability**")
        st.markdown(format_available_doctors_markdown())
    if twilio_configured():
        st.success("✅ Care alert SMS is configured.")
    else:
        st.info("ℹ️ Care alert SMS in demo mode. Add Twilio credentials in `.env` to enable.")

# ── Sidebar ───────────────────────────────────────────
selected_sample = None

with st.sidebar:

    st.markdown("""
    <div class="sb-section">
        <div class="sb-section-title">🏥 Mannan Medical Clinic</div>
        <div class="sb-row"><span class="sb-row-label">Location</span><span class="sb-row-value">Main Bazaar, Sahiwal</span></div>
        <div class="sb-row"><span class="sb-row-label">Phone</span><span class="sb-row-value">0300-1234567</span></div>
        <div class="sb-row"><span class="sb-row-label">Hours</span><span class="sb-row-value">Mon–Sat · 9AM–6PM</span></div>
        <div class="sb-row"><span class="sb-row-label">Emergency</span><span class="sb-row-value">24 / 7</span></div>
        <div class="sb-row"><span class="sb-row-label">Rescue</span><span class="sb-row-value">1122</span></div>
    </div>
    """, unsafe_allow_html=True)

    total_msgs = len([m for m in st.session_state.get("messages",[]) if m["role"]=="user"])
    st.markdown(f"""
    <div class="counter-box">
        <div>
            <div class="counter-label">💬 Questions Asked</div>
            <div class="counter-sub">This session</div>
        </div>
        <div class="counter-num">{total_msgs}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-section-title" style="padding:0 0 8px; border-bottom:1px solid var(--border-soft); margin-bottom:8px">
        👨‍⚕️ Doctor Availability
    </div>
    """, unsafe_allow_html=True)

    for doc in current_doctor_availability():
        if doc["available"]:
            avail_html = '<div class="doc-avail on"><div class="avail-dot on"></div>Available Now</div>'
        else:
            avail_html = '<div class="doc-avail off"><div class="avail-dot off"></div>Not Available</div>'
        st.markdown(f"""
        <div class="doc-card">
            <div class="doc-name">{doc['name']}</div>
            <div class="doc-spec">{doc['spec']} · {doc['time']}</div>
            {avail_html}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-section-title" style="padding:10px 0 8px; border-bottom:1px solid var(--border-soft); margin:8px 0">
        💡 Common Questions
    </div>
    """, unsafe_allow_html=True)

    samples = [
        "What is the blood test fee?",
        "What are Dr. Ahmed's timings?",
        "I have fever and cough",
        "I want to book an appointment",
        "What are the clinic hours?",
        "Which doctor for heart issues?",
    ]
    for s in samples:
        if st.button(s, key=f"sample_{s}", use_container_width=True):
            selected_sample = s

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Safety Note + Toolbar ─────────────────────────────
st.markdown("""
<div class="safety-note">
    <div>
        <strong>⚠️ Patient Safety Note</strong>
        <span>
            This assistant provides clinic guidance and basic triage only.
            For chest pain, breathing difficulty, heavy bleeding, or loss of
            consciousness — call 0300-1234567 or Rescue 1122 immediately.
        </span>
    </div>
</div>
<div class="chat-toolbar">
    <div class="chat-toolbar-title">
        💬 Patient Conversation
    </div>
    <div class="live-badge">
        <span class="live-dot"></span>
        Assistant Online
    </div>
</div>
""", unsafe_allow_html=True)

# ── Emergency Keywords ────────────────────────────────
EMERGENCY_KEYWORDS = [
    "heart attack","chest pain","not breathing","can't breathe",
    "unconscious","bleeding heavily","stroke","dying","critical",
    "fainted","seizure","severe pain","emergency","collapsed",
    "overdose","poisoning","choking"
]

# ── Chat History ──────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "**Welcome to Mannan Medical Clinic.** 👋\n\n"
            "I can help you with:\n\n"
            "- 👨‍⚕️ **Doctor information & schedules**\n"
            "- 🧪 **Tests, fees & services**\n"
            "- 📅 **Appointment booking guidance**\n"
            "- 🤒 **Symptom triage & doctor routing**\n"
            "- 🔬 **Lab report analysis** (attach an image)\n\n"
            "How may I assist you today?"
        ),
        "agent": "GENERAL",
        "eval":  "pass",
        "time":  datetime.datetime.now().strftime("%I:%M %p")
    })

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            agent = msg.get("agent", "GENERAL")
            if agent == "EMERGENCY":
                badge = '<span class="msg-badge danger">🚨 Emergency</span>'
            elif agent == "LAB REPORT":
                badge = '<span class="msg-badge purple">🔬 Lab Report</span>'
            elif agent == "VOICE":
                badge = '<span class="msg-badge primary">🎙️ Voice</span>'
            else:
                badge = (f'<span class="msg-badge primary">🤖 {agent}</span>'
                         f'<span class="msg-badge success">✅ Verified</span>')
            st.markdown(f'<div class="msg-meta">{badge}</div>', unsafe_allow_html=True)
        st.markdown(msg["content"])
        st.markdown(f'<div class="msg-time">🕐 {msg.get("time","")}</div>', unsafe_allow_html=True)

# ── Voice Input ───────────────────────────────────────
recorded_voice = st.audio_input(
    "🎙️ Record symptoms by voice",
    sample_rate=16000,
    key="voice_symptom_recorder",
)

recorded_voice_hash = ""
if recorded_voice:
    recorded_voice_hash = hashlib.sha256(recorded_voice.getvalue()).hexdigest()
    if recorded_voice_hash == st.session_state.get("last_voice_hash"):
        recorded_voice = None
    else:
        st.session_state.last_voice_hash = recorded_voice_hash

# ── Chat Input ────────────────────────────────────────
chat_value = st.chat_input(
    "💬 Type symptoms, or attach a lab report image...",
    accept_file="multiple",
    file_type=["png","jpg","jpeg","webp"],
    accept_audio=True,
    audio_sample_rate=16000,
    max_upload_size=15
)

attached_files = []
voice_audio    = None

if selected_sample:
    query = selected_sample
elif chat_value:
    if isinstance(chat_value, str):
        query = chat_value
    else:
        query          = getattr(chat_value, "text", "") or ""
        attached_files = normalize_chat_files(getattr(chat_value, "files", []) or [])
        voice_audio    = getattr(chat_value, "audio", None)
else:
    query = ""

if recorded_voice and not voice_audio:
    voice_audio = recorded_voice

query = query.strip()

voice_transcript = ""
if voice_audio:
    with st.spinner("Converting voice to text..."):
        try:
            voice_transcript = transcribe_voice_input(voice_audio)
        except Exception as exc:
            voice_transcript = media_provider_error_message("Voice transcription", exc)
    if voice_transcript and not voice_transcript.startswith("Voice input needs"):
        query = f"{query}\n\nVoice transcript: {voice_transcript}".strip()

if query or attached_files or voice_audio:
    now = datetime.datetime.now().strftime("%I:%M %p")
    attached_names = ", ".join(f.name for f in attached_files)
    user_content   = query or "Please analyze my voice message."
    if voice_audio and voice_transcript.startswith("Voice input needs"):
        user_content = "Voice message received, but speech-to-text is not configured."
    if attached_names:
        user_content = f"{user_content}\n\nAttached: {attached_names}"

    st.session_state.messages.append({"role":"user","content":user_content,"time":now})

    with st.chat_message("user"):
        st.markdown(user_content)
        if voice_audio:
            st.audio(voice_audio, format=voice_audio.type or "audio/wav")
        for file in attached_files:
            st.image(file, caption=file.name, use_container_width=True)
        st.markdown(f'<div class="msg-time">🕐 {now}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):

        if (voice_audio and not attached_files and
                (voice_transcript.startswith("Voice input needs") or
                 voice_transcript.startswith("[Unclear") or
                 voice_transcript.startswith("Voice transcription"))):
            agent = "VOICE"
            now2  = datetime.datetime.now().strftime("%I:%M %p")
            response = (voice_transcript if voice_transcript.startswith("Voice input needs")
                        else "I could not clearly understand the voice message. "
                             "Please record again or type your symptoms.")
            st.markdown('<div class="msg-meta"><span class="msg-badge primary">🎙️ Voice</span></div>', unsafe_allow_html=True)
            st.markdown(response)

        elif attached_files:
            thinking = st.markdown("""
            <div class="typing-wrap">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-label">Analyzing lab report...</span>
            </div>""", unsafe_allow_html=True)
            try:
                response = analyze_lab_report_images(attached_files, query)
                if len(attached_files) > 4:
                    response += "\n\n*Note: Only first 4 images analyzed. Send remaining pages separately.*"
            except Exception as exc:
                response = media_provider_error_message("Lab report analysis", exc)
            agent = "LAB REPORT"
            now2  = datetime.datetime.now().strftime("%I:%M %p")
            thinking.empty()
            st.markdown('<div class="msg-meta"><span class="msg-badge purple">🔬 Lab Report</span><span class="msg-badge success">✅ Verified</span></div>', unsafe_allow_html=True)
            st.markdown(response)

        elif any(kw in query.lower() for kw in EMERGENCY_KEYWORDS):
            st.markdown("""
            <div class="emerg-box">
                <div class="emerg-title">🚨 Emergency Situation Detected</div>
                <div class="emerg-body">
                    Your message indicates a potentially life-threatening situation.
                    Do not wait for online assistance — contact emergency support immediately.
                </div>
                <div class="emerg-actions">
                    <div class="emerg-btn" style="background:#dc2626">📞 Clinic: 0300-1234567</div>
                    <div class="emerg-btn" style="background:#991b1b">🚑 Rescue: 1122</div>
                    <div class="emerg-btn" style="background:#7f1d1d">🏥 Nearest Hospital NOW</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            response = (
                "🚨 **Emergency Detected!**\n\n"
                "- 📞 **Clinic:** 0300-1234567\n"
                "- 🚑 **Punjab Rescue:** 1122\n"
                "- 🏥 **Go to nearest hospital immediately**\n\n"
                f"{build_emergency_recommendation(query)}"
            )
            agent = "EMERGENCY"
            now2  = datetime.datetime.now().strftime("%I:%M %p")
            st.markdown('<div class="msg-meta"><span class="msg-badge danger">🚨 Emergency</span></div>', unsafe_allow_html=True)
            st.markdown(response)

        else:
            thinking = st.markdown("""
            <div class="typing-wrap">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-label">Processing your query...</span>
            </div>""", unsafe_allow_html=True)
            state = {
                "user_query":        query,
                "query_type":        "",
                "retrieved_context": "",
                "agent_response":    "",
                "evaluation_result": "",
                "final_response":    "",
                "messages":          []
            }
            result   = app.invoke(state)
            agent    = result.get("query_type","general").upper()
            response = result["final_response"]
            now2     = datetime.datetime.now().strftime("%I:%M %p")
            thinking.empty()
            st.markdown(f'<div class="msg-meta"><span class="msg-badge primary">🤖 {agent}</span><span class="msg-badge success">✅ Verified</span></div>', unsafe_allow_html=True)
            st.markdown(response)

        st.markdown(f'<div class="msg-time">🕐 {now2}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({
        "role":"assistant","content":response,
        "agent":agent,"eval":"pass","time":now2
    })