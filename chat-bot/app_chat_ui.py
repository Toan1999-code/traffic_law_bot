from flask import Flask, request, jsonify, render_template_string, send_file
from Step3_rag_traffic_law_bot import ask_traffic_law_bot
import webbrowser
import threading
import io
import os
import tempfile
import subprocess

import torch
from scipy.io import wavfile
from transformers import pipeline, VitsModel, AutoTokenizer
from dotenv import load_dotenv
from openai import OpenAI

# ==============================
#  LOAD ENV + OPENAI CLIENT
# ==============================

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

oa_client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
chat_history = []

# ==============================
#  LOAD STT / TTS MODELS
# ==============================

"""
STT: dùng Whisper fine-tuned cho tiếng Việt.
Bạn có thể đổi sang model khác nếu muốn, ví dụ:
- "namphungdn134/whisper-small-vi"
- "kiendt/whisper-small-vivos"
"""
STT_MODEL_NAME = "namphungdn134/whisper-small-vi"

stt_pipeline = pipeline(
    "automatic-speech-recognition",
    model=STT_MODEL_NAME
)

"""
TTS: dùng facebook/mms-tts-vie (Tiếng Việt, VITS).
"""

TTS_MODEL_NAME = "facebook/mms-tts-vie"

tts_model = VitsModel.from_pretrained(TTS_MODEL_NAME)
tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_NAME)


# ==============================
#  LLM SPEECH NORMALIZER
# ==============================

def normalize_for_tts_vi(text: str) -> str:
    """
    Dùng LLM viết lại câu trả lời thành phiên bản dễ đọc bằng giọng nói tiếng Việt:
    - Giữ nguyên nghĩa.
    - Viết số, tiền, điều/khoản, ngày tháng... thành CHỮ tiếng Việt.
    - Bỏ bớt ký hiệu khó đọc như '/', '.', '-' trong số.
    - Chia câu dài thành các câu ngắn hơn nếu cần.
    Nếu có lỗi khi gọi LLM, fallback về text gốc.
    """
    if not text.strip():
        return text

    system_msg = (
        "Bạn là bộ chuyển đổi văn bản sang dạng dễ đọc bằng giọng nói tiếng Việt.\n"
        "Nhiệm vụ của bạn:\n"
        "- Giữ nguyên nghĩa câu trả lời.\n"
        "- Viết toàn bộ số, số tiền, số điều/khoản, năm tháng... thành CHỮ tiếng Việt.\n"
        "- Hạn chế dùng kí hiệu như '/', '.', '-' trong số; thay bằng cách đọc tự nhiên.\n"
        "- Có thể chia câu dài thành 2–3 câu ngắn hơn cho dễ đọc.\n"
        "- Không thêm nhận xét, không thêm lời giải thích; chỉ trả về phiên bản văn bản đã chỉnh sửa."
    )

    try:
        resp = oa_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": text},
            ],
        )
        out = resp.choices[0].message.content.strip()
        return out or text
    except Exception:
        # Nếu có lỗi (hết quota / mạng / ...), dùng luôn text gốc
        return text


# ==============================
#  HTML + JS UI
# ==============================

HTML_PAGE = r"""
{% raw %}
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <title>Traffic Law Assistant - Chat</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
  :root {
    --bg: #f3f4f6;
    --bg-elevated: #ffffff;
    --bg-sidebar: #f9fafb;
    --accent: #16a34a;
    --accent-soft: rgba(22, 163, 74, 0.12);
    --accent-strong: #15803d;
    --text: #111827;
    --text-soft: #6b7280;
    --border: #e5e7eb;
    --user-bubble: #1d4ed8;
    --assistant-bubble: #f9fafb;
    --font: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }

  * {
    box-sizing: border-box;
  }

  body {
    margin: 0;
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
  }

  .app-root {
    display: flex;
    height: 100vh;
    width: 100vw;
    overflow: hidden;
  }

  /* ========== SIDEBAR (CHAT LIST) ========== */

  .sidebar {
    width: 260px;
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border);
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .sidebar-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 4px;
  }

  .sidebar-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-soft);
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .sidebar-title span.icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 999px;
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    font-size: 11px;
  }

  .btn-new-chat {
    border-radius: 999px;
    border: 1px solid #d1d5db;
    padding: 4px 10px;
    font-size: 12px;
    background: #ecfdf3;
    color: #15803d;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }

  .btn-new-chat span.icon {
    font-size: 13px;
  }

  .btn-new-chat:hover {
    border-color: #22c55e;
    background: #bbf7d0;
  }

  .chat-list {
    flex: 1;
    overflow-y: auto;
    padding-right: 4px;
    margin-top: 4px;
  }

  .chat-item {
    padding: 8px 10px;
    border-radius: 10px;
    font-size: 13px;
    color: var(--text-soft);
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    border: 1px solid transparent;
    margin-bottom: 4px;
  }

  .chat-item-icon {
    width: 18px;
    height: 18px;
    border-radius: 999px;
    background: #e5e7eb;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    color: #6b7280;
  }

  .chat-item.active {
    background: #e0f2fe;
    border-color: #38bdf8;
    color: #0f172a;
  }

  .chat-item:hover {
    background: #f3f4f6;
    border-color: #d1d5db;
  }

  .chat-item-title {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .sidebar-footer {
    font-size: 11px;
    color: var(--text-soft);
    border-top: 1px solid var(--border);
    padding-top: 6px;
    margin-top: 4px;
  }

  /* ========== MAIN PANE (CHAT) ========== */

  .main-pane {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 12px;
    max-width: 980px;
    margin: 0 auto;
    width: 100%;
  }

  .app-header {
    padding: 12px 16px;
    border-radius: 16px;
    background: #ffffff;
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
  }

  .app-title {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .app-title h1 {
    font-size: 18px;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 8px;
    color: #0f172a;
  }

  .app-title h1 span.logo {
    width: 24px;
    height: 24px;
    border-radius: 8px;
    background: radial-gradient(circle at 30% 30%, #22c55e, #16a34a);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    font-weight: 700;
    color: #ecfdf3;
  }

  .chat-title-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .app-title p {
    margin: 0;
    font-size: 12px;
    color: var(--text-soft);
  }

  .current-chat-title {
    font-size: 13px;
    color: var(--text-soft);
  }

  .app-header-right {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--text-soft);
  }

  .pill {
    border-radius: 999px;
    padding: 4px 10px;
    border: 1px solid #e5e7eb;
    background: #f9fafb;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  .pill-dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: var(--accent);
    box-shadow: 0 0 6px rgba(22, 163, 74, 0.7);
  }

  .pill-badge {
    border-radius: 999px;
    padding: 2px 8px;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    background: #ecfdf3;
    border: 1px solid #86efac;
    color: #166534;
    font-weight: 600;
  }

  .chat-container {
    flex: 1;
    margin-top: 10px;
    margin-bottom: 10px;
    border-radius: 16px;
    background: #ffffff;
    border: 1px solid var(--border);
    box-shadow: 0 4px 18px rgba(148, 163, 184, 0.15);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .chat-messages {
    flex: 1;
    padding: 16px;
    overflow-y: auto;
    scroll-behavior: smooth;
    background: #f9fafb;
  }

  .message-row {
    display: flex;
    margin-bottom: 12px;
  }

  .message-row.user {
    justify-content: flex-end;
  }

  .message-row.assistant {
    justify-content: flex-start;
  }

  .message-bubble {
    max-width: 80%;
    padding: 10px 12px;
    border-radius: 16px;
    font-size: 14px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-wrap: break-word;
    box-shadow: 0 4px 10px rgba(148, 163, 184, 0.35);
    border: 1px solid #e5e7eb;
  }

  .message-row.user .message-bubble {
    background: linear-gradient(135deg, #2563eb, #22c55e);
    color: #f9fafb;
    border-radius: 16px 4px 16px 16px;
  }

  .message-row.assistant .message-bubble {
    background: #ffffff;
    color: var(--text);
    border-radius: 4px 16px 16px 16px;
  }

  .message-meta {
    font-size: 11px;
    color: var(--text-soft);
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .message-role {
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .message-time {
    opacity: 0.8;
  }

  .chat-footer {
    padding: 10px;
    border-top: 1px solid var(--border);
    background: #ffffff;
  }

  .input-row {
    display: flex;
    gap: 8px;
    align-items: flex-end;
  }

  textarea#user-input {
    flex: 1;
    resize: none;
    min-height: 44px;
    max-height: 120px;
    border-radius: 12px;
    border: 1px solid #d1d5db;
    background: #ffffff;
    color: var(--text);
    padding: 8px 10px;
    font-family: var(--font);
    font-size: 14px;
    outline: none;
  }

  textarea#user-input::placeholder {
    color: var(--text-soft);
    opacity: 0.7;
  }

  button#send-btn {
    border: none;
    border-radius: 999px;
    padding: 10px 16px;
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: #f9fafb;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 8px 18px rgba(22, 163, 74, 0.35);
    transition: transform 0.08s ease, box-shadow 0.08s ease, background 0.1s ease;
    white-space: nowrap;
  }

  button#send-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    box-shadow: none;
  }

  button#send-btn:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 10px 24px rgba(22, 163, 74, 0.55);
  }

  button#send-btn span.icon {
    font-size: 16px;
  }

  .footer-bar {
    margin-top: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    color: var(--text-soft);
  }

  .status-text {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 999px;
    background: var(--accent-soft);
    border: 1px solid var(--accent);
    animation: pulse 1.2s infinite;
  }

  @keyframes pulse {
    0% { transform: scale(1); opacity: 0.7; }
    50% { transform: scale(1.3); opacity: 1; }
    100% { transform: scale(1); opacity: 0.7; }
  }

  .history-controls {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .small-btn {
    border-radius: 999px;
    border: 1px solid #d1d5db;
    background: #f9fafb;
    color: var(--text-soft);
    padding: 3px 8px;
    font-size: 11px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }

  .small-btn:hover {
    border-color: #9ca3af;
    color: #111827;
  }

  .small-btn-header {
    border-radius: 999px;
    border: 1px solid #d1d5db;
    background: #f9fafb;
    color: var(--text-soft);
    padding: 2px 6px;
    font-size: 11px;
    cursor: pointer;
  }

  .small-btn-header:hover {
    border-color: #9ca3af;
    color: #111827;
  }

  .small-btn-header.danger {
    border-color: #fecaca;
    color: #b91c1c;
    background: #fef2f2;
  }

  .small-btn-header.danger:hover {
    border-color: #fca5a5;
    background: #fee2e2;
    color: #7f1d1d;
  }

  .message-row.assistant.thinking .message-bubble {
    font-style: italic;
    opacity: 0.85;
  }

  /* ====== NÚT RECORD MIC ====== */
  .icon-btn {
    border-radius: 999px;
    border: 1px solid #d1d5db;
    background: #f9fafb;
    color: #111827;
    padding: 8px;
    font-size: 18px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 38px;
  }

  .icon-btn:hover {
    border-color: #9ca3af;
    background: #e5e7eb;
  }

  .icon-btn.recording {
    background: #fee2e2;
    border-color: #f87171;
    color: #b91c1c;
  }

  @media (max-width: 840px) {
    .app-root {
      flex-direction: column;
    }
    .sidebar {
      width: 100%;
      border-right: none;
      border-bottom: 1px solid var(--border);
      flex: 0 0 auto;
    }
    .main-pane {
      padding: 8px;
    }
    .app-header {
      flex-direction: column;
      align-items: flex-start;
    }
    .app-header-right {
      width: 100%;
      justify-content: space-between;
    }
  }
  </style>
</head>
<body>
  <div class="app-root">
    <!-- SIDEBAR -->
    <aside class="sidebar">
      <div class="sidebar-header">
        <div class="sidebar-title">
          <span class="icon">💬</span>
          <span>Danh sách cuộc trò chuyện</span>
        </div>
        <button class="btn-new-chat" id="new-chat-btn">
          <span class="icon">＋</span>
          <span>Cuộc trò chuyện mới</span>
        </button>
      </div>
      <div id="chat-list" class="chat-list"></div>
      <div class="sidebar-footer">
        <div>⚖️ Trợ lý pháp lý giao thông</div>
        <div>Luật 36/2024/QH15 · NĐ 168/2024/NĐ-CP</div>
      </div>
    </aside>

    <!-- MAIN PANE -->
    <div class="main-pane">
      <header class="app-header">
        <div class="app-title">
          <h1>
            <span class="logo">TL</span>
            Traffic Law Assistant
          </h1>
          <div class="chat-title-row">
            <p id="current-chat-title" class="current-chat-title">
              Cuộc trò chuyện mới
            </p>
            <button id="rename-chat-btn" class="small-btn-header" title="Đổi tên cuộc trò chuyện">
              ✏️
            </button>
            <button id="delete-chat-btn" class="small-btn-header danger" title="Xóa cuộc trò chuyện này">
              🗑
            </button>
          </div>
        </div>
        <div class="app-header-right">
          <div class="pill">
            <span class="pill-dot"></span>
            <span>RAG: ChromaDB + OpenAI</span>
          </div>
          <span class="pill-badge">Local Client</span>
        </div>
      </header>

      <main class="chat-container">
        <div id="chat-messages" class="chat-messages"></div>

        <footer class="chat-footer">
          <div class="input-row">
            <textarea
              id="user-input"
              rows="1"
              placeholder="Nhập câu hỏi về luật giao thông... (Nhấn Enter để gửi, Shift+Enter để xuống dòng)"
            ></textarea>
            <!-- Nút ghi âm -->
            <button id="record-btn" class="icon-btn" title="Ghi âm câu hỏi (giọng nói tiếng Việt)">🎙</button>
            <button id="send-btn">
              <span class="icon">➤</span>
              Gửi
            </button>
          </div>
          <div class="footer-bar">
            <div class="status-text" id="status-text">
              <span class="status-dot"></span>
              <span>Sẵn sàng trả lời dựa trên dữ liệu Luật &amp; Nghị định đã nạp.</span>
            </div>
            <div class="history-controls">
              <button class="small-btn" id="tts-btn">🔊 Đọc câu trả lời</button>
              <button class="small-btn" id="clear-btn">🧹 Xóa lịch sử cuộc trò chuyện hiện tại</button>
            </div>
          </div>
        </footer>
      </main>
    </div>
  </div>

  <script>
    // ==============================
    //  CONFIG
    // ==============================

    const API_URL = "/chat";
    const STORAGE_KEY = "trafficLawConversations";
    const STORAGE_ACTIVE_KEY = "trafficLawActiveConversationId";

    const chatMessagesEl = document.getElementById("chat-messages");
    const userInputEl = document.getElementById("user-input");
    const sendBtnEl = document.getElementById("send-btn");
    const statusTextEl = document.getElementById("status-text");
    const clearBtnEl = document.getElementById("clear-btn");
    const chatListEl = document.getElementById("chat-list");
    const newChatBtn = document.getElementById("new-chat-btn");
    const currentChatTitleEl = document.getElementById("current-chat-title");
    const renameChatBtnEl = document.getElementById("rename-chat-btn");
    const deleteChatBtnEl = document.getElementById("delete-chat-btn");

    const recordBtnEl = document.getElementById("record-btn");
    const ttsBtnEl = document.getElementById("tts-btn");

    let isSending = false;
    let voiceMode = true; // đang bật chế độ voice

    // conversations: array of { id, title, messages: [{role, text, timestamp}] }
    let conversations = [];
    let activeConversationId = null;

    // ==============================
    //  STORAGE HELPERS
    // ==============================

    function loadConversations() {
      try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw) {
          const parsed = JSON.parse(raw);
          if (Array.isArray(parsed)) {
            conversations = parsed;
          }
        }
        const active = localStorage.getItem(STORAGE_ACTIVE_KEY);
        if (active) {
          activeConversationId = active;
        }
      } catch (e) {
        console.warn("Không thể load localStorage:", e);
      }

      if (conversations.length === 0) {
        const conv = createNewConversationObject("Cuộc trò chuyện mới");
        conversations.push(conv);
        activeConversationId = conv.id;
      }

      if (!activeConversationId || !conversations.find(c => c.id === activeConversationId)) {
        activeConversationId = conversations[0].id;
      }
    }

    function saveConversations() {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
        if (activeConversationId) {
          localStorage.setItem(STORAGE_ACTIVE_KEY, activeConversationId);
        }
      } catch (e) {
        console.warn("Không thể lưu localStorage:", e);
      }
    }

    function createNewConversationObject(defaultTitle) {
      const id = "conv_" + Date.now() + "_" + Math.floor(Math.random() * 100000);
      return {
        id,
        title: defaultTitle || "Cuộc trò chuyện mới",
        messages: []
      };
    }

    function getActiveConversation() {
      return conversations.find(c => c.id === activeConversationId) || null;
    }

    // ==============================
    //  RENDER SIDEBAR + CHAT
    // ==============================

    function renderChatList() {
      chatListEl.innerHTML = "";
      conversations.forEach(conv => {
        const div = document.createElement("div");
        div.className = "chat-item" + (conv.id === activeConversationId ? " active" : "");
        div.dataset.id = conv.id;

        const icon = document.createElement("div");
        icon.className = "chat-item-icon";
        icon.textContent = "💬";

        const title = document.createElement("div");
        title.className = "chat-item-title";
        title.textContent = conv.title || "Cuộc trò chuyện";

        div.appendChild(icon);
        div.appendChild(title);
        chatListEl.appendChild(div);

        div.addEventListener("click", () => {
          if (conv.id !== activeConversationId) {
            activeConversationId = conv.id;
            saveConversations();
            renderChatList();
            renderCurrentConversation();
          }
        });
      });
    }

    function renderCurrentConversation() {
      const conv = getActiveConversation();
      chatMessagesEl.innerHTML = "";

      if (!conv) return;

      currentChatTitleEl.textContent = conv.title || "Cuộc trò chuyện";

      conv.messages.forEach(msg => {
        addMessageToDOM(msg.role, msg.text, msg.timestamp, { skipStore: true });
      });

      chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
    }

    // ==============================
    //  MESSAGE RENDER
    // ==============================

    function addMessageToDOM(role, text, timestamp, options = {}) {
      const row = document.createElement("div");
      row.classList.add("message-row", role);
      if (options.thinking) {
        row.classList.add("thinking");
      }

      const bubble = document.createElement("div");
      bubble.classList.add("message-bubble");

      const meta = document.createElement("div");
      meta.classList.add("message-meta");

      const roleSpan = document.createElement("span");
      roleSpan.classList.add("message-role");
      roleSpan.textContent = role === "user" ? "Bạn" : "Assistant";

      const timeSpan = document.createElement("span");
      timeSpan.classList.add("message-time");
      const dateObj = timestamp ? new Date(timestamp) : new Date();
      timeSpan.textContent = dateObj.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });

      meta.appendChild(roleSpan);
      meta.appendChild(document.createTextNode("•"));
      meta.appendChild(timeSpan);

      bubble.appendChild(meta);
      bubble.appendChild(document.createTextNode(text));

      row.appendChild(bubble);
      chatMessagesEl.appendChild(row);
      chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;

      if (!options.skipStore) {
        const conv = getActiveConversation();
        if (conv) {
          conv.messages.push({
            role,
            text,
            timestamp: dateObj.toISOString()
          });
          if (role === "user" && conv.messages.length === 1) {
            conv.title = text.slice(0, 40) + (text.length > 40 ? "…" : "");
            currentChatTitleEl.textContent = conv.title;
          }
          saveConversations();
          renderChatList();
        }
      }

      return row;
    }

    function setStatus(text) {
      if (!statusTextEl) return;
      const span = statusTextEl.querySelector("span:nth-child(2)");
      if (span) span.textContent = text;
    }

    function setSending(state) {
      isSending = state;
      sendBtnEl.disabled = state;
      userInputEl.disabled = state;
      if (state) {
        setStatus("Assistant đang xử lý câu hỏi...");
      } else {
        setStatus("Sẵn sàng trả lời dựa trên dữ liệu Luật & Nghị định đã nạp.");
      }
    }

    // ==============================
    //  RENAME / DELETE CONVERSATION
    // ==============================

    function renameActiveConversation() {
      const conv = getActiveConversation();
      if (!conv) return;

      const currentTitle = conv.title || "Cuộc trò chuyện";
      const newTitle = window.prompt("Nhập tên mới cho cuộc trò chuyện:", currentTitle);

      if (newTitle === null) {
        return; // user bấm Cancel
      }

      const trimmed = newTitle.trim();
      if (!trimmed) {
        alert("Tên cuộc trò chuyện không được để trống.");
        return;
      }

      conv.title = trimmed;
      currentChatTitleEl.textContent = conv.title;
      saveConversations();
      renderChatList();
    }

    function deleteActiveConversation() {
      const conv = getActiveConversation();
      if (!conv) return;

      const confirmMsg = `Bạn có chắc chắn muốn xóa cuộc trò chuyện:\n"${conv.title || "Cuộc trò chuyện"}"?`;
      if (!window.confirm(confirmMsg)) {
        return;
      }

      const idx = conversations.findIndex(c => c.id === conv.id);
      if (idx !== -1) {
        conversations.splice(idx, 1);
      }

      if (conversations.length === 0) {
        const newConv = createNewConversationObject("Cuộc trò chuyện mới");
        conversations.push(newConv);
        activeConversationId = newConv.id;
      } else {
        const newIndex = Math.max(0, idx - 1);
        activeConversationId = conversations[newIndex].id;
      }

      saveConversations();
      renderChatList();
      renderCurrentConversation();
      const active = getActiveConversation();
      addWelcomeIfEmpty(active);

      // Reset context phía server
      fetch("/reset_chat", { method: "POST" }).catch(console.error);
    }

    // ==============================
    //  SEND MESSAGE
    // ==============================

    async function sendMessage() {
      if (isSending) return;

      const text = userInputEl.value.trim();
      if (!text) return;

      const conv = getActiveConversation();
      if (!conv) return;

      addMessageToDOM("user", text);
      userInputEl.value = "";
      userInputEl.style.height = "auto";

      const thinkingRow = addMessageToDOM("assistant", "Đang suy nghĩ...", null, { thinking: true });

      try {
        setSending(true);

        const response = await fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text }),
        });

        if (!response.ok) {
          throw new Error("Lỗi mạng hoặc server (" + response.status + ")");
        }

        const data = await response.json();
        const reply = data.reply || "(Không nhận được nội dung trả lời từ server.)";

        const bubble = thinkingRow.querySelector(".message-bubble");
        if (bubble) {
          while (bubble.childNodes.length > 1) {
            bubble.removeChild(bubble.lastChild);
          }
          bubble.appendChild(document.createTextNode(reply));
        }

        thinkingRow.classList.remove("thinking");

        const dateObj = new Date();
        const conv2 = getActiveConversation();
        if (conv2) {
          conv2.messages.push({
            role: "assistant",
            text: reply,
            timestamp: dateObj.toISOString()
          });
          saveConversations();
        }

        chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
      } catch (err) {
        console.error(err);
        const bubble = thinkingRow.querySelector(".message-bubble");
        if (bubble) {
          while (bubble.childNodes.length > 1) {
            bubble.removeChild(bubble.lastChild);
          }
          bubble.appendChild(
            document.createTextNode(
              "Đã xảy ra lỗi khi kết nối tới server: " + err.message
            )
          );
        }
        thinkingRow.classList.remove("thinking");
      } finally {
        setSending(false);
      }
    }

    // ==============================
    //  VOICE: RECORD & STT
    // ==============================

    let mediaRecorder = null;
    let recordedChunks = [];
    let isRecording = false;

    async function initMediaRecorder() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Trình duyệt của bạn không hỗ trợ ghi âm (getUserMedia).");
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) {
            recordedChunks.push(event.data);
          }
        };

        mediaRecorder.onstop = async () => {
          const blob = new Blob(recordedChunks, { type: "audio/webm" });
          recordedChunks = [];

          try {
            const formData = new FormData();
            formData.append("audio", blob, "recording.webm");

            const res = await fetch("/stt", {
              method: "POST",
              body: formData
            });

            let data = {};
            try {
              data = await res.json();
            } catch (e) {
              data = {};
            }

            if (!res.ok) {
              const msg = data.error || ("STT lỗi: " + res.status);
              throw new Error(msg);
            }

            if (data.text) {
              // 1) hiển thị text vào ô input
              userInputEl.value = data.text;
              userInputEl.dispatchEvent(new Event("input"));

              // 2) nếu đang ở chế độ voice, tự động gửi luôn
              if (voiceMode) {
                await sendMessage();
              }
            } else {
              alert("Không nhận được kết quả nhận diện giọng nói.");
            }
          } catch (err) {
            console.error(err);
            alert("Lỗi khi gửi audio tới server: " + err.message);
          } finally {
            recordBtnEl.classList.remove("recording");
            recordBtnEl.textContent = "🎙";
            isRecording = false;
          }
        };
      } catch (err) {
        console.error(err);
        alert("Không thể truy cập micro: " + err.message);
      }
    }

    if (recordBtnEl) {
      recordBtnEl.addEventListener("click", async () => {
        if (!mediaRecorder) {
          await initMediaRecorder();
          if (!mediaRecorder) return;
        }

        if (!isRecording) {
          recordedChunks = [];
          mediaRecorder.start();
          isRecording = true;
          recordBtnEl.classList.add("recording");
          recordBtnEl.textContent = "⏹";
        } else {
          mediaRecorder.stop();
        }
      });
    }

    // ==============================
    //  VOICE: TTS (ĐỌC CÂU TRẢ LỜI)
    // ==============================

    async function speakLastAssistantMessage() {
      const conv = getActiveConversation();
      if (!conv || !conv.messages.length) {
        alert("Chưa có câu trả lời nào để đọc.");
        return;
      }
      const lastAssistant = [...conv.messages].reverse().find(m => m.role === "assistant");
      if (!lastAssistant) {
        alert("Chưa có câu trả lời nào để đọc.");
        return;
      }

      try {
        const res = await fetch("/tts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: lastAssistant.text })
        });

        if (!res.ok) {
          throw new Error("TTS lỗi: " + res.status);
        }

        const arrayBuffer = await res.arrayBuffer();
        const blob = new Blob([arrayBuffer], { type: "audio/wav" });
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.play();
      } catch (err) {
        console.error(err);
        alert("Không phát được âm thanh: " + err.message);
      }
    }

    if (ttsBtnEl) {
      ttsBtnEl.addEventListener("click", speakLastAssistantMessage);
    }

    // ==============================
    //  EVENTS
    // ==============================

    userInputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    userInputEl.addEventListener("input", () => {
      userInputEl.style.height = "auto";
      userInputEl.style.height = userInputEl.scrollHeight + "px";
    });

    sendBtnEl.addEventListener("click", () => {
      sendMessage();
    });

    clearBtnEl.addEventListener("click", () => {
      const conv = getActiveConversation();
      if (conv) {
        conv.messages = [];
        saveConversations();
      }
      chatMessagesEl.innerHTML = "";
      addWelcomeIfEmpty(conv);

      // reset context server
      fetch("/reset_chat", { method: "POST" }).catch(console.error);
    });

    newChatBtn.addEventListener("click", () => {
      const conv = createNewConversationObject("Cuộc trò chuyện mới");
      conversations.unshift(conv);
      activeConversationId = conv.id;
      saveConversations();
      renderChatList();
      renderCurrentConversation();
      addWelcomeIfEmpty(conv);

      // reset context server cho cuộc trò chuyện mới
      fetch("/reset_chat", { method: "POST" }).catch(console.error);
    });

    renameChatBtnEl.addEventListener("click", () => {
      renameActiveConversation();
    });

    deleteChatBtnEl.addEventListener("click", () => {
      deleteActiveConversation();
    });

    function addWelcomeIfEmpty(conv) {
      const c = conv || getActiveConversation();
      if (!c) return;
      if (c.messages.length === 0) {
        const welcomeText =
          "Xin chào! Tôi là trợ lý pháp lý giao thông.\n" +
          "Bạn có thể hỏi tôi về Luật Trật tự, an toàn giao thông đường bộ 2024 " +
          "và Nghị định 168/2024/NĐ-CP (xử phạt, trừ/khôi phục điểm giấy phép lái xe).\n\n" +
          "Ví dụ:\n" +
          "- \"Không đội mũ bảo hiểm khi đi xe máy bị phạt bao nhiêu?\"\n" +
          "- \"Giấy phép lái xe hạng A1 được điều khiển xe gì?\"\n" +
          "- \"Quên bật đèn trong hầm đường bộ có bị phạt không?\"";

        addMessageToDOM("assistant", welcomeText);
      }
    }

    // ==============================
    //  INIT
    // ==============================

    window.addEventListener("load", () => {
      loadConversations();
      renderChatList();
      renderCurrentConversation();

      const active = getActiveConversation();
      addWelcomeIfEmpty(active);
    });
  </script>
</body>
</html>
{% endraw %}
"""

# ==============================
#  ROUTES
# ==============================

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"reply": "Vui lòng nhập câu hỏi hợp lệ."}), 400

    try:
        reply = ask_traffic_law_bot(
            question=message,
            top_k=8,
            chat_history=chat_history,
        )

        # Update chat history (server-side, dùng cho rewrite_question_node)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"Lỗi server: {str(e)}"}), 500


@app.route("/reset_chat", methods=["POST"])
def reset_chat():
    """Reset lịch sử chat phía server (cho LangGraph)."""
    global chat_history
    chat_history = []
    return jsonify({"status": "ok"})


# ==============================
#  STT ENDPOINT
# ==============================

@app.route("/stt", methods=["POST"])
def stt():
    file = request.files.get("audio")
    if file is None:
        return jsonify({"error": "Thiếu file audio."}), 400

    # lưu file webm tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_webm:
        tmp_webm.write(file.read())
        webm_path = tmp_webm.name

    # chuyển sang wav
    wav_path = webm_path.replace(".webm", ".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except Exception as e:
        return jsonify({"error": f"FFmpeg convert error: {str(e)}"}), 500

    try:
        # chạy Whisper trên file WAV
        result = stt_pipeline(wav_path)
        text = result.get("text", "").strip()
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": f"STT error: {str(e)}"}), 500
    finally:
        # dọn file
        if os.path.exists(webm_path):
            os.remove(webm_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)


# ==============================
#  TTS ENDPOINT
# ==============================

@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return "Missing text", 400

    try:
        # Bước 1: chuẩn hóa text cho dễ đọc bằng giọng nói
        norm_text = normalize_for_tts_vi(text)

        # Bước 2: đưa vào MMS TTS
        inputs = tts_tokenizer(norm_text, return_tensors="pt")
        with torch.no_grad():
            output = tts_model(**inputs).waveform

        # output: (1, num_samples)
        audio = output.squeeze().cpu().numpy()
        buffer = io.BytesIO()
        wavfile.write(buffer, tts_model.config.sampling_rate, audio)
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="audio/wav",
            as_attachment=False,
            download_name="tts.wav"
        )
    except Exception as e:
        return f"TTS error: {str(e)}", 500


def open_browser():
    webbrowser.open("http://127.0.0.1:8000", new=1)


if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    print("Server đang chạy tại http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, debug=True)
