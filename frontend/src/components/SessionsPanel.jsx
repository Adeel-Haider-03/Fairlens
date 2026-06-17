import { useState, useEffect } from 'react'

const API = 'http://localhost:8000'

export default function SessionsPanel({ onLoad, onClose }) {
  const [sessions, setSessions] = useState([])
  const [loading,  setLoading]  = useState(true)
  const [renaming, setRenaming] = useState(null)   // session_id being renamed
  const [newName,  setNewName]  = useState('')
  const [deleting, setDeleting] = useState(null)

  useEffect(() => { fetchSessions() }, [])

  const fetchSessions = async () => {
    setLoading(true)
    try {
      const res  = await fetch(`${API}/api/sessions`)
      const data = await res.json()
      setSessions(data)
    } catch {
      setSessions([])
    } finally {
      setLoading(false)
    }
  }

  const loadSession = async (session_id) => {
    try {
      const res     = await fetch(`${API}/api/sessions/${session_id}`)
      const results = await res.json()
      onLoad(results)
      onClose()
    } catch (e) {
      alert('Failed to load session: ' + e.message)
    }
  }

  const deleteSession = async (session_id) => {
    setDeleting(session_id)
    try {
      await fetch(`${API}/api/sessions/${session_id}`, { method: 'DELETE' })
      setSessions(s => s.filter(x => x.session_id !== session_id))
    } catch {}
    setDeleting(null)
  }

  const renameSession = async (session_id) => {
    try {
      await fetch(`${API}/api/sessions/${session_id}/rename`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newName }),
      })
      setSessions(s => s.map(x => x.session_id === session_id ? { ...x, session_name: newName } : x))
    } catch {}
    setRenaming(null)
    setNewName('')
  }

  const formatDate = (iso) => {
    try { return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) }
    catch { return iso }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
         style={{ background: 'rgba(10,14,26,0.85)', backdropFilter: 'blur(8px)' }}>
      <div className="card-glow w-full max-w-2xl max-h-[80vh] flex flex-col gap-0 p-0 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-ink-700">
          <div>
            <h3 className="font-display font-bold text-white text-base">Saved Sessions</h3>
            <p className="text-xs text-ink-400 font-body mt-0.5">Load any previous run instantly — no retraining needed</p>
          </div>
          <button onClick={onClose} className="btn-ghost text-xs px-3 py-1.5">✕ Close</button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {loading && (
            <div className="flex items-center justify-center py-16 gap-3">
              <div className="w-6 h-6 rounded-full border-2 border-cyan/30 border-t-cyan animate-spin" />
              <span className="text-ink-400 font-body text-sm">Loading sessions...</span>
            </div>
          )}

          {!loading && sessions.length === 0 && (
            <div className="text-center py-16">
              <div className="w-12 h-12 rounded-2xl bg-ink-700 flex items-center justify-center mx-auto mb-3">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4a5f96" strokeWidth="1.5" strokeLinecap="round">
                  <path d="M3 3h18v18H3zM8 12h8M12 8v8"/>
                </svg>
              </div>
              <p className="text-ink-500 font-body text-sm">No sessions saved yet.</p>
              <p className="text-ink-600 text-xs mt-1">Complete a training run to save results here.</p>
            </div>
          )}

          {sessions.map(s => (
            <div key={s.session_id}
                 className="bg-ink-800 border border-ink-700 rounded-xl p-4 hover:border-ink-600 transition-all">
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                  {renaming === s.session_id ? (
                    <div className="flex gap-2 mb-2">
                      <input autoFocus value={newName}
                        onChange={e => setNewName(e.target.value)}
                        onKeyDown={e => { if(e.key==='Enter') renameSession(s.session_id); if(e.key==='Escape') setRenaming(null) }}
                        className="input-field text-sm py-1.5 flex-1"
                        placeholder="Session name..." />
                      <button onClick={() => renameSession(s.session_id)} className="btn-primary text-xs px-3 py-1.5">Save</button>
                      <button onClick={() => setRenaming(null)} className="btn-ghost text-xs px-3 py-1.5">✕</button>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-display font-bold text-sm text-white truncate">{s.session_name}</span>
                      <button onClick={() => { setRenaming(s.session_id); setNewName(s.session_name) }}
                              className="text-ink-600 hover:text-ink-300 transition-colors flex-shrink-0">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                          <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/>
                        </svg>
                      </button>
                    </div>
                  )}
                  <p className="text-[11px] text-ink-500 font-body mb-2">{formatDate(s.created_at)}</p>
                  <div className="flex flex-wrap gap-1.5">
                    <span className="tag border-cyan/20 text-cyan/70 bg-cyan/5 text-[10px]">
                      target: {s.target}
                    </span>
                    <span className="tag border-amber/20 text-amber/70 bg-amber/5 text-[10px]">
                      protected: {s.protected}
                    </span>
                    {s.models?.map(m => (
                      <span key={m} className="tag border-ink-600 text-ink-400 text-[10px]">{m}</span>
                    ))}
                  </div>
                </div>

                {/* Metrics + actions */}
                <div className="flex flex-col items-end gap-2 flex-shrink-0">
                  {s.best_accuracy && (
                    <div className="text-right">
                      <p className="text-[10px] text-ink-500 font-display uppercase tracking-wider">Best Acc</p>
                      <p className="font-mono font-bold text-emerald text-sm">{(s.best_accuracy*100).toFixed(1)}%</p>
                    </div>
                  )}
                  {s.best_spd !== null && s.best_spd !== undefined && (
                    <div className="text-right">
                      <p className="text-[10px] text-ink-500 font-display uppercase tracking-wider">Best SPD</p>
                      <p className={`font-mono font-bold text-sm ${Math.abs(s.best_spd) < 0.05 ? 'text-emerald' : 'text-amber'}`}>
                        {s.best_spd.toFixed(4)}
                      </p>
                    </div>
                  )}
                </div>
              </div>

              <div className="flex gap-2 mt-3 pt-3 border-t border-ink-700">
                <button onClick={() => loadSession(s.session_id)}
                        className="btn-primary text-xs py-1.5 flex-1">
                  ⚡ Load Results
                </button>
                <button
                  onClick={() => deleteSession(s.session_id)}
                  disabled={deleting === s.session_id}
                  className="btn-ghost text-xs py-1.5 px-3 text-rose/70 hover:border-rose/40 hover:text-rose">
                  {deleting === s.session_id ? '...' : 'Delete'}
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        {sessions.length > 0 && (
          <div className="px-6 py-3 border-t border-ink-700">
            <p className="text-[11px] text-ink-600 font-body text-center">
              {sessions.length} session{sessions.length !== 1 ? 's' : ''} saved · Results persist across server restarts
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
