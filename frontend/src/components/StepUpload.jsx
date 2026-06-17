import { useState, useRef } from 'react'

const API = 'http://localhost:8000'

export default function StepUpload({ onNext }) {
  const [dragging, setDragging]   = useState(false)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)
  const [preview, setPreview]     = useState(null)
  const fileRef = useRef()

  const handleFile = async (file) => {
    if (!file || !file.name.endsWith('.csv')) {
      setError('Please upload a CSV file.')
      return
    }
    setError(null)
    setLoading(true)

    const form = new FormData()
    form.append('file', file)

    try {
      // 30 second timeout so it never silently hangs
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 30000)

      const res = await fetch(`${API}/api/upload`, {
        method: 'POST',
        body: form,
        signal: controller.signal,
      })
      clearTimeout(timeout)

      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Upload failed')
      setPreview(data)
    } catch (e) {
      if (e.name === 'AbortError') {
        setError('Upload timed out. Is the backend running at http://localhost:8000?')
      } else if (e.message.includes('fetch')) {
        setError('Cannot connect to backend. Make sure uvicorn is running on port 8000.')
      } else {
        setError(e.message)
      }
    } finally {
      setLoading(false)
    }
  }

  const onDrop = (e) => {
    e.preventDefault(); setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  return (
    <div className="fade-up max-w-2xl mx-auto space-y-6">
      <div className="text-center mb-8">
        <h2 className="font-display font-bold text-2xl text-white mb-2">Upload Your Dataset</h2>
        <p className="text-ink-400 font-body text-sm">Supports CSV files. Missing values (?/space) are handled automatically.</p>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => fileRef.current.click()}
        className={`card-glow cursor-pointer transition-all duration-300 flex flex-col items-center justify-center py-16 gap-4 ${
          dragging ? 'border-cyan bg-cyan/10 scale-[1.01]' : 'border-ink-600 hover:border-cyan/50 hover:bg-ink-700/30'
        }`}
      >
        <input ref={fileRef} type="file" accept=".csv" className="hidden"
               onChange={e => handleFile(e.target.files[0])} />

        {loading ? (
          <>
            <div className="w-12 h-12 rounded-full border-2 border-cyan/30 border-t-cyan animate-spin" />
            <span className="font-display text-sm text-cyan">Processing CSV...</span>
          </>
        ) : (
          <>
            <div className={`w-16 h-16 rounded-2xl flex items-center justify-center transition-all ${
              dragging ? 'bg-cyan/20 scale-110' : 'bg-ink-700'
            }`}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke={dragging ? '#00E5FF' : '#4a5f96'} strokeWidth="1.5" strokeLinecap="round">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
              </svg>
            </div>
            <div className="text-center">
              <p className="font-display font-bold text-white text-sm">
                {dragging ? 'Drop to upload' : 'Drag & drop or click to browse'}
              </p>
              <p className="font-body text-xs text-ink-500 mt-1">CSV files only · No size limit</p>
            </div>
          </>
        )}
      </div>

      {error && (
        <div className="card border-rose/30 bg-rose/5 text-rose text-sm font-body flex items-center gap-2">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>
          </svg>
          {error}
        </div>
      )}

      {/* Preview */}
      {preview && (
        <div className="card-glow space-y-4 fade-up">
          {/* Stats row */}
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-display font-bold text-white text-sm">{preview.filename}</h3>
              <p className="font-body text-xs text-ink-400 mt-0.5">
                {preview.shape.rows.toLocaleString()} rows · {preview.shape.columns} columns
              </p>
            </div>
            <div className="tag border-emerald/30 text-emerald bg-emerald/5">
              <svg width="10" height="10" viewBox="0 0 12 12" fill="none">
                <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Loaded
            </div>
          </div>

          {/* Column chips */}
          <div>
            <p className="label">Columns detected</p>
            <div className="flex flex-wrap gap-2">
              {preview.columns.map(col => (
                <span key={col.name}
                  className={`tag text-[11px] ${col.is_numeric
                    ? 'border-cyan/20 text-cyan/80 bg-cyan/10'
                    : 'border-amber/20 text-amber/80 bg-amber/5'}`}>
                  {col.name}
                  <span className="opacity-50">{col.is_numeric ? 'num' : 'cat'}</span>
                </span>
              ))}
            </div>
          </div>

          {/* Data table preview */}
          <div className="overflow-x-auto rounded-xl border border-ink-700">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="bg-ink-700/60">
                  {preview.columns.slice(0,6).map(c => (
                    <th key={c.name} className="px-3 py-2 text-left text-ink-300 font-bold border-b border-ink-700 whitespace-nowrap">
                      {c.name}
                    </th>
                  ))}
                  {preview.columns.length > 6 && <th className="px-3 py-2 text-ink-500 border-b border-ink-700">+{preview.columns.length - 6} more</th>}
                </tr>
              </thead>
              <tbody>
                {preview.preview.map((row, i) => (
                  <tr key={i} className="border-b border-ink-800 hover:bg-ink-700/40 transition-colors">
                    {preview.columns.slice(0,6).map(c => (
                      <td key={c.name} className="px-3 py-2 text-ink-300 whitespace-nowrap">
                        {String(row[c.name]).slice(0,20)}
                      </td>
                    ))}
                    {preview.columns.length > 6 && <td className="px-3 py-2 text-ink-400">...</td>}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Dataset warnings */}
          {preview.warnings && preview.warnings.length > 0 && (
            <div className="space-y-2">
              {preview.warnings.map((w, i) => (
                <div key={i} className={`flex items-start gap-3 rounded-xl p-3 border text-xs font-body ${
                  w.level === 'error'
                    ? 'border-rose-500/50 bg-rose-500/20 text-rose-200'
                    : 'border-amber-500/50 bg-amber-500/20 text-amber-200'
                }`}>
                  <span className="text-lg flex-shrink-0 mt-0.5">{w.level === 'error' ? '🔴' : '⚠️'}</span>
                  <p>{w.msg}</p>
                </div>
              ))}
            </div>
          )}

          <button onClick={() => onNext(preview)} className="btn-primary w-full">
            Continue to Configuration →
          </button>
        </div>
      )}
    </div>
  )
}
