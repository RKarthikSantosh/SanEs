import { useState, useRef } from 'react'
import { uploadDocument } from '../api.js'

// Supported file types matching the backend SUPPORTED_EXTENSIONS
const ACCEPT = '.pdf,.docx,.txt,.md'

export default function UploadPanel({ onUploaded }) {
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  async function handleFile(file) {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const result = await uploadDocument(file)
      onUploaded(result)
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {/* Drop zone */}
      <div
        className={`dropzone${dragging ? ' active' : ''}`}
        onClick={() => !loading && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        role="button"
        tabIndex={0}
        aria-label="Upload document"
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          style={{ display: 'none' }}
          onChange={(e) => handleFile(e.target.files[0])}
        />
        <div style={{ fontSize: 22, marginBottom: 6 }}>↑</div>
        {loading
          ? <span>Processing...</span>
          : <span>Drop a file or click to browse</span>}
        <div style={{ fontSize: 11, marginTop: 4, color: 'var(--text-muted)' }}>PDF, DOCX, TXT, MD</div>
      </div>

      {/* Error */}
      {error && (
        <div style={{ fontSize: 12, color: '#ef4444', padding: '6px 10px', background: 'var(--bg-elevated)', borderRadius: 6, border: '1px solid #3f1212' }}>
          {error}
        </div>
      )}
    </div>
  )
}
