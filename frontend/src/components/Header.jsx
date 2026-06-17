export default function Header({ onOpenSessions }) {
  return (
    <header className="border-b border-ink-700/60 backdrop-blur-sm sticky top-0 z-50 bg-ink-900/80">
      <div className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center"
               style={{ background: 'linear-gradient(135deg,#00E5FF22,#00E5FF11)', border: '1px solid rgba(0,229,255,0.3)' }}>
            <span className="font-display font-bold text-cyan text-sm">FL</span>
          </div>
          <div>
            <span className="font-display font-bold text-white text-lg tracking-tight">Fair</span>
            <span className="font-display font-bold glow-text text-lg tracking-tight">Lens</span>
          </div>
          <div className="hidden sm:block h-4 w-px bg-ink-600 ml-1" />
          <span className="hidden sm:block font-body text-xs text-ink-400">AI Bias Detection & Mitigation</span>
        </div>

        <div className="flex items-center gap-2">
          {/* Sessions button */}
          <button onClick={onOpenSessions}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-ink-600 text-ink-300 hover:border-cyan/40 hover:text-cyan transition-all text-xs font-display font-bold">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/>
            </svg>
            Saved Sessions
          </button>
          <span className="tag border-cyan/20 text-cyan/80 bg-cyan/5 text-[10px]">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald inline-block" />
            AIF360 Powered
          </span>
        </div>
      </div>
    </header>
  )
}
