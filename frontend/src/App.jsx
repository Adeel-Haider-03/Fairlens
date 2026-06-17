import { useState } from 'react'
import Header from './components/Header'
import StepBar from './components/StepBar'
import StepUpload from './components/StepUpload'
import StepConfigure from './components/StepConfigure'
import StepModels from './components/StepModels'
import StepTraining from './components/StepTraining'
import StepResults from './components/StepResults'
import SessionsPanel from './components/SessionsPanel'

const STEPS = [
  { id: 1, label: 'Upload',    short: '01' },
  { id: 2, label: 'Configure', short: '02' },
  { id: 3, label: 'Models',    short: '03' },
  { id: 4, label: 'Training',  short: '04' },
  { id: 5, label: 'Results',   short: '05' },
]

export default function App() {
  const [step, setStep] = useState(1)
  const [showSessions, setShowSessions] = useState(false)

  const [uploadData,  setUploadData]  = useState(null)
  const [config,      setConfig]      = useState(null)
  const [modelConfig, setModelConfig] = useState(null)
  const [jobId,       setJobId]       = useState(null)
  const [results,     setResults]     = useState(null)

  const next = () => setStep(s => Math.min(s + 1, 5))
  const back = () => setStep(s => Math.max(s - 1, 1))
  const goTo = (n) => setStep(n)

  // Load a session directly from the sessions panel → jump straight to results
  const handleLoadSession = (sessionResults) => {
    setResults(sessionResults)
    // Reconstruct minimal config from saved results for display
    const cfg = sessionResults.config || {}
    setConfig({
      target_column:       cfg.target_column      || '—',
      protected_attribute: cfg.protected_attribute || '—',
      privileged_value:    cfg.privileged_value    || '—',
      favorable_value:     cfg.favorable_value     || null,
      protected_threshold: cfg.protected_threshold || null,
      feature_columns:     cfg.feature_columns     || [],
    })
    setModelConfig({
      models:           cfg.models           || [],
      mitigation_steps: cfg.mitigation_steps || [],
      smote_variants:   cfg.smote_variants   || [],
    })
    setStep(5)
  }

  const handleReset = () => {
    setStep(1); setUploadData(null); setConfig(null)
    setModelConfig(null); setJobId(null); setResults(null)
  }

  return (
    <div className="min-h-screen grid-bg">
      <Header onOpenSessions={() => setShowSessions(true)} />

      {showSessions && (
        <SessionsPanel
          onLoad={handleLoadSession}
          onClose={() => setShowSessions(false)}
        />
      )}

      <div className="max-w-6xl mx-auto px-4 pb-16">
        <StepBar steps={STEPS} current={step} onGo={goTo} />
        <div className="mt-8">
          {step === 1 && <StepUpload onNext={(d) => { setUploadData(d); next() }} />}
          {step === 2 && uploadData && (
            <StepConfigure uploadData={uploadData} onNext={(c) => { setConfig(c); next() }} onBack={back} />
          )}
          {step === 3 && (
            <StepModels onNext={(m) => { setModelConfig(m); next() }} onBack={back} />
          )}
          {step === 4 && (
            <StepTraining
              uploadData={uploadData} config={config} modelConfig={modelConfig}
              onDone={(jid, res) => { setJobId(jid); setResults(res); next() }}
              onBack={back}
            />
          )}
          {step === 5 && results && (
            <StepResults results={results} config={config} modelConfig={modelConfig} onReset={handleReset} />
          )}
        </div>
      </div>
    </div>
  )
}
