import { useState, useRef, useEffect } from 'react'
import { askQuestion } from '../api.js'

// Renders inline citation numbers like [1]
function renderAnswer(text) {
  const parts = text.split(/(\[\d+\])/g)
  return parts.map((part, i) => {
    const m = part.match(/^\[(\d+)\]$/)
    return m ? <span key={i} className="cite">[{m[1]}]</span> : part
  })
}

function Message({ msg }) {
  const isUser = msg.role === 'user'
  const hasCitations = msg.citations && msg.citations.length > 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <div className={isUser ? 'bubble-user' : 'bubble-assistant'}>
        {isUser ? msg.content : renderAnswer(msg.content)}
      </div>

      {/* Show sources */}
      {hasCitations && (
        <CitationList citations={msg.citations} />
      )}
    </div>
  )
}

function CitationList({ citations }) {
  const [open, setOpen] = useState(false)
  const count = citations.length

  return (
    <div style={{ alignSelf: 'flex-start', maxWidth: '85%' }}>
      <button
        className="btn"
        style={{ fontSize: 11, padding: '3px 10px', marginTop: 4 }}
        onClick={() => setOpen(o => !o)}
      >
        {open ? 'Hide' : 'Show'} {count} cited source{count !== 1 ? 's' : ''}
      </button>
      {open && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 6 }}>
          {citations.map((c, i) => (
            <div key={i} className="passage-card">
              <div className="passage-meta">
                [{c.number}] {c.filename} — page {c.page}
                {c.score !== undefined && c.score !== null && (
                  <span style={{ marginLeft: 8 }}>score: {Number(c.score).toFixed(4)}</span>
                )}
              </div>
              <div style={{ wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}>
                {c.text || c.snippet}
              </div>
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
  const [topK, setTopK] = useState('auto')
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
      const topNParam = topK === 'auto' ? 'auto' : parseInt(topK, 10)
      const data = await askQuestion(q, topNParam)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer || 'No answer returned.',
        citations: data.citations || [],
      }])
    } catch (e) {
      const detail = e.response?.data?.detail
      const errorMsg = Array.isArray(detail)
        ? detail.map(d => d.msg || JSON.stringify(d)).join(', ')
        : (typeof detail === 'object' && detail !== null ? JSON.stringify(detail) : (detail || e.message))
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${errorMsg}`,
        citations: [],
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
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, padding: '10px 16px 12px', background: 'var(--bg-surface)' }}>
        {/* Dynamic Top-K selection toolbar */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-muted)' }}>
            <span>Top-k:</span>
            <select
              id="top-k-select"
              value={topK}
              onChange={e => setTopK(e.target.value)}
              className="input"
              style={{ width: 'auto', padding: '2px 8px', fontSize: 11, height: 24, cursor: 'pointer' }}
            >
              <option value="auto">Auto (dynamic)</option>
              <option value="3">3</option>
              <option value="5">5</option>
              <option value="7">7</option>
              <option value="10">10</option>
            </select>
          </div>
          {topK === 'auto' ? (
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              Dynamic context sizing by query & relevance
            </span>
          ) : (
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              Retrieves top {topK} candidates
            </span>
          )}
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
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
    </div>
  )
}
