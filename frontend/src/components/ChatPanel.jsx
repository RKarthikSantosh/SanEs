import { useState, useRef, useEffect } from 'react'
import { askQuestion } from '../api.js'

// Renders inline citation numbers like [1]
function renderAnswer(text) {
  const parts = text.split(/(\[\d+\])/g)
  return parts.map((part, i) => {
    const m = part.match(/^\[(\d+)\]$/)
    return m ? <sup key={i} className="cite">{m[1]}</sup> : part
  })
}

function Message({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <div className={isUser ? 'bubble-user' : 'bubble-assistant'}>
        {isUser ? msg.content : renderAnswer(msg.content)}
      </div>

      {/* Collapsed passages from the pipeline */}
      {msg.passages && msg.passages.length > 0 && (
        <PassageList passages={msg.passages} />
      )}
    </div>
  )
}

function PassageList({ passages }) {
  const [open, setOpen] = useState(false)
  return (
    <div style={{ alignSelf: 'flex-start', maxWidth: '85%' }}>
      <button
        className="btn"
        style={{ fontSize: 11, padding: '3px 10px', marginTop: 4 }}
        onClick={() => setOpen(o => !o)}
      >
        {open ? 'Hide' : 'Show'} {passages.length} source{passages.length !== 1 ? 's' : ''}
      </button>
      {open && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 6 }}>
          {passages.map((p, i) => (
            <div key={i} className="passage-card">
              <div className="passage-meta">
                [{i + 1}] {p.filename} — page {p.page}
                {p.score !== undefined && <span style={{ marginLeft: 8 }}>score: {p.score.toFixed(4)}</span>}
              </div>
              <div>{p.text}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="bubble-assistant" style={{ color: 'var(--text-muted)', fontSize: 13 }}>
      Retrieving and generating...
    </div>
  )
}

export default function ChatPanel({ hasDocuments }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  // Scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function send() {
    const q = input.trim()
    if (!q || loading) return
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: q }])
    setLoading(true)
    try {
      const data = await askQuestion(q)
      // data.answer contains the grounded response and
      // data.passages contains retrieved chunks
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer || 'No answer returned.',
        passages: data.passages || [],
      }])
    } catch (e) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${e.response?.data?.detail || e.message}`,
        passages: [],
      }])
    } finally {
      setLoading(false)
    }
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Message list */}
      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 14, padding: '16px 20px' }}>
        {messages.length === 0 && !loading && (
          <div style={{ margin: 'auto', textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
            {hasDocuments
              ? 'Ask a question about your documents.'
              : 'Upload a document to get started.'}
          </div>
        )}
        {messages.map((m, i) => <Message key={i} msg={m} />)}
        {loading && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>

      <hr className="divider" />

      {/* Input bar */}
      <div style={{ display: 'flex', gap: 8, padding: '12px 16px', background: 'var(--bg-surface)' }}>
        <textarea
          id="chat-input"
          className="input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={hasDocuments ? 'Ask a question...' : 'Upload a document first'}
          disabled={!hasDocuments || loading}
          rows={1}
          style={{ resize: 'none', lineHeight: 1.5, overflowY: 'hidden' }}
          onInput={e => {
            // Auto-grow textarea up to 5 lines
            e.target.style.height = 'auto'
            e.target.style.height = Math.min(e.target.scrollHeight, 100) + 'px'
          }}
        />
        <button
          id="chat-send"
          className="btn btn-primary"
          onClick={send}
          disabled={!hasDocuments || !input.trim() || loading}
          style={{ whiteSpace: 'nowrap', alignSelf: 'flex-end' }}
        >
          Send
        </button>
      </div>
    </div>
  )
}
