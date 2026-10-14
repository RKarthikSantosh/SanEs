import { useState, useRef, useEffect, useCallback } from 'react'
import { uploadDocument, getUploadStatus } from '../api.js'

// Supported file types matching the backend SUPPORTED_EXTENSIONS
const ACCEPT = '.pdf,.docx,.txt,.md'

// Ordered processing steps for the progress indicator
const STEPS = [
  { key: 'queued',     label: 'Queued' },
  { key: 'extracting', label: 'Extracting text' },
  { key: 'chunking',   label: 'Chunking pages' },
  { key: 'embedding',  label: 'Generating embeddings' },
  { key: 'storing',    label: 'Storing in database' },
  { key: 'done',       label: 'Complete' },
]

function StepIndex(stepKey) {
  const idx = STEPS.findIndex(s => s.key === stepKey)
  return idx === -1 ? 0 : idx
}

// Processing status indicator
function ProcessingStatus({ task }) {
  if (!task) return null

  const currentIdx = StepIndex(task.step)
  const isFailed = task.status === 'failed'
  const isDone = task.status === 'completed'

  return (
    <div style={{
      padding: '10px 12px',
      borderRadius: 8,
      background: 'var(--bg-elevated)',
      border: `1px solid ${isFailed ? '#ef4444' : 'var(--border-subtle)'}`,
    }}>
      {/* Filename */}
      <div style={{
        fontSize: 12, fontWeight: 500, color: 'var(--text-primary)',
        marginBottom: 8, wordBreak: 'break-all',
      }}>
        {task.filename}
      </div>

      {/* Step list */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {STEPS.map((step, i) => {
          let state = 'pending'
          if (isFailed && i === currentIdx) state = 'failed'
          else if (i < currentIdx) state = 'done'
          else if (i === currentIdx && !isDone) state = 'active'
          else if (isDone) state = 'done'

          const dotColor = {
            done: '#22c55e',
            active: '#3b82f6',
            failed: '#ef4444',
            pending: 'var(--border)',
          }[state]

          return (
            <div key={step.key} style={{
              display: 'flex', alignItems: 'center', gap: 8, fontSize: 11,
              color: state === 'active' ? 'var(--text-primary)'
                   : state === 'done' ? 'var(--text-secondary)'
                   : state === 'failed' ? '#ef4444'
                   : 'var(--text-muted)',
            }}>
              {/* Animated dot for active step */}
              <span style={{
                width: 6, height: 6, borderRadius: '50%',
                background: dotColor, flexShrink: 0,
                animation: state === 'active' ? 'pulse-dot 1.2s ease-in-out infinite' : 'none',
              }} />
              <span>{step.label}</span>
              {state === 'active' && (
                <span className="spinner" style={{ marginLeft: 'auto' }} />
              )}
            </div>
          )
        })}
      </div>

      {/* Status message */}
      {task.message && (
        <div style={{
          fontSize: 11, color: isFailed ? '#ef4444' : 'var(--text-muted)',
          marginTop: 8, borderTop: '1px solid var(--border-subtle)',
          paddingTop: 6,
        }}>
          {task.message}
        </div>
      )}
    </div>
  )
}

export default function UploadPanel({ onUploaded }) {
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [task, setTask] = useState(null)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)
  const pollRef = useRef(null)

  // Cleanup polling
  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [])

  const startPolling = useCallback((taskId) => {
    if (pollRef.current) clearInterval(pollRef.current)

    pollRef.current = setInterval(async () => {
      try {
        const status = await getUploadStatus(taskId)
        setTask(status)

        if (status.status === 'completed') {
          clearInterval(pollRef.current)
          pollRef.current = null
          if (status.result) onUploaded(status.result)

          setTimeout(() => setTask(null), 2000)
        } else if (status.status === 'failed') {
          clearInterval(pollRef.current)
          pollRef.current = null
        }
      } catch {
        // keep polling even if one request fails
      }
    }, 800)
  }, [onUploaded])

  async function handleFile(file) {
    if (!file) return
    setUploading(true)
    setError(null)
    setTask(null)
    try {
      const data = await uploadDocument(file)

      setTask({
        status: 'processing',
        step: 'queued',
        message: 'Upload received, starting processing...',
        filename: data.filename,
      })
      setUploading(false)
      startPolling(data.task_id)
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
      setUploading(false)
    } finally {
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }

  const isProcessing = uploading || (task && task.status === 'processing')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {/* Drop zone */}
      <div
        className={`dropzone${dragging ? ' active' : ''}`}
        onClick={() => !isProcessing && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        role="button"
        tabIndex={0}
        aria-label="Upload document"
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
        style={{ opacity: isProcessing ? 0.5 : 1, pointerEvents: isProcessing ? 'none' : 'auto' }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          style={{ display: 'none' }}
          onChange={(e) => handleFile(e.target.files[0])}
        />
        <div style={{ fontSize: 22, marginBottom: 6 }}>↑</div>
        {uploading
          ? <span>Uploading file...</span>
          : <span>Drop a file or click to browse</span>}
        <div style={{ fontSize: 11, marginTop: 4, color: 'var(--text-muted)' }}>PDF, DOCX, TXT, MD</div>
      </div>

      {/* Processing progress indicator */}
      {task && <ProcessingStatus task={task} />}

      {/* Error */}
      {error && (
        <div style={{ fontSize: 12, color: '#ef4444', padding: '6px 10px', background: 'var(--bg-elevated)', borderRadius: 6, border: '1px solid #3f1212' }}>
          {error}
        </div>
      )}
    </div>
  )
}
