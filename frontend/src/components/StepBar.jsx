export default function StepBar({ steps, current, onGo }) {
  return (
    <div className="flex items-center justify-center pt-8 gap-0">
      {steps.map((step, i) => {
        const done   = current > step.id
        const active = current === step.id
        return (
          <div key={step.id} className="flex items-center">
            <button
              onClick={() => done && onGo(step.id)}
              className={`flex items-center gap-2.5 px-3 py-2 rounded-xl transition-all duration-300 ${
                done ? 'cursor-pointer hover:bg-ink-700/50' : 'cursor-default'
              }`}
            >
              {/* Number circle */}
              <div className={`step-dot ${
                active
                  ? 'bg-cyan text-ink-900 pulse-glow'
                  : done
                  ? 'bg-emerald/20 text-emerald border border-emerald/40'
                  : 'bg-ink-700 text-ink-400 border border-ink-600'
              }`}>
                {done
                  ? <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                      <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  : <span>{step.short}</span>
                }
              </div>
              <span className={`font-display text-xs font-bold uppercase tracking-wider hidden sm:block transition-colors ${
                active ? 'text-cyan' : done ? 'text-emerald/80' : 'text-ink-500'
              }`}>
                {step.label}
              </span>
            </button>

            {/* Connector line */}
            {i < steps.length - 1 && (
              <div className={`h-px w-8 mx-1 transition-all duration-500 ${
                current > step.id ? 'bg-emerald/50' : 'bg-ink-700'
              }`} />
            )}
          </div>
        )
      })}
    </div>
  )
}
