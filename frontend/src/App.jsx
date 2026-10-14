import { useState, useEffect } from 'react'
import UploadPanel from './components/UploadPanel.jsx'
import ChatPanel from './components/ChatPanel.jsx'
import { checkHealth } from './api.js'

// Store theme in localStorage
function getInitialTheme() {
  return localStorage.getItem('sanes-theme') || 'light'
}

export default function App() {
  const [theme, setTheme] = useState(getInitialTheme)
  const [backendOk, setBackendOk] = useState(null)
  const [documents, setDocuments] = useState([])

  // Apply theme
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('sanes-theme', theme)
  }, [theme])

  // Check backend health
  useEffect(() => {
    checkHealth().then(setBackendOk).catch(() => setBackendOk(false))
  }, [])

  function toggleTheme() {
    setTheme(t => t === 'light' ? 'dark' : 'light')
  }

  function handleUploaded(result) {
    setDocuments(prev => {
      // Replace entry if same filename was re-uploaded
      const filtered = prev.filter(d => d.filename !== result.filename)
      return [result, ...filtered]
    })
  }

  const hasDocuments = documents.length > 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: 'var(--bg-base)' }}>
      {/* Top bar */}
      <header style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 20px', height: 48, borderBottom: '1px solid var(--border-subtle)',
        background: 'var(--bg-surface)', flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>
            SanEs
          </span>
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Document Intelligence</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {/* Backend status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 12, color: 'var(--text-muted)' }}>
            <span className={`dot ${backendOk === null ? 'dot-yellow' : backendOk ? 'dot-green' : 'dot-red'}`} />
            {backendOk === null ? 'Connecting...' : backendOk ? 'Backend connected' : 'Backend offline'}
          </div>

          {/* Theme toggle */}
          <button
            id="theme-toggle"
            className="btn"
            onClick={toggleTheme}
            aria-label="Toggle theme"
            style={{ padding: '4px 10px', fontSize: 12 }}
          >
            {theme === 'light' ? 'Dark' : 'Light'}
          </button>
        </div>
      </header>

      {/* Main layout: sidebar + chat */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>

        {/* Sidebar */}
        <aside style={{
          width: 260, flexShrink: 0, borderRight: '1px solid var(--border-subtle)',
          background: 'var(--bg-surface)', display: 'flex', flexDirection: 'column', overflow: 'hidden',
        }}>
          <div style={{ padding: '14px 16px 10px', borderBottom: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginBottom: 10 }}>
              Upload Document
            </div>
            <UploadPanel onUploaded={handleUploaded} />
          </div>

          {/* Document list */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '10px 16px' }}>
            {documents.length > 0 && (
              <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginBottom: 8 }}>
                Indexed ({documents.length})
              </div>
            )}
            {documents.map((doc, i) => (
              <div key={i} style={{
                padding: '8px 10px', borderRadius: 6, marginBottom: 4,
                background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)',
              }}>
                <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)', wordBreak: 'break-all' }}>
                  {doc.filename}
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                  {doc.pages} page{doc.pages !== 1 ? 's' : ''} &middot; {doc.chunks} chunks
                </div>
              </div>
            ))}
            {documents.length === 0 && (
              <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>No documents indexed yet.</p>
            )}
          </div>
        </aside>

        {/* Chat area */}
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', background: 'var(--bg-base)' }}>
          <ChatPanel hasDocuments={hasDocuments} />
        </main>
      </div>
    </div>
  )
}
