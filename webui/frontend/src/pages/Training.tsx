import { useState, useEffect, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import { listJobs, getJob, getMetrics, createTrainJob, cancelJob, listRecipes, listConfigs, writeConfigToDisk, type ConfigRecord, type Job, type MetricPoint } from '../api'

const STATUS_COLORS: Record<string, string> = {
  queued: 'text-gray-400',
  running: 'text-blue-400',
  completed: 'text-green-400',
  failed: 'text-red-400',
  cancelled: 'text-yellow-400',
}

const STATUS_DOT: Record<string, string> = {
  queued: 'bg-gray-500',
  running: 'bg-blue-400 animate-pulse',
  completed: 'bg-green-400',
  failed: 'bg-red-400',
  cancelled: 'bg-yellow-400',
}

const METRIC_COLORS = ['#60a5fa', '#f59e0b', '#34d399', '#f87171', '#a78bfa', '#fb923c']

function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`flex items-center gap-1.5 text-xs ${STATUS_COLORS[status] ?? 'text-gray-400'}`}>
      <span className={`w-1.5 h-1.5 rounded-full inline-block ${STATUS_DOT[status] ?? 'bg-gray-500'}`} />
      {status}
    </span>
  )
}

// Build chart data: merge multiple metric series by step
function buildChartData(metrics: Record<string, MetricPoint[]>, keys: string[]) {
  const byStep: Record<number, Record<string, number>> = {}
  for (const key of keys) {
    for (const { step, value } of (metrics[key] ?? [])) {
      byStep[step] = { ...byStep[step], [key]: value }
    }
  }
  return Object.entries(byStep)
    .sort(([a], [b]) => Number(a) - Number(b))
    .map(([step, vals]) => ({ step: Number(step), ...vals }))
}

function MetricsChart({
  metrics, title, metricKeys,
}: {
  metrics: Record<string, MetricPoint[]>
  title: string
  metricKeys: string[]
}) {
  const available = metricKeys.filter(k => (metrics[k]?.length ?? 0) > 0)
  if (available.length === 0) return <p className="text-gray-500 text-sm">No {title} data yet.</p>
  const data = buildChartData(metrics, available)
  return (
    <div className="mb-4">
      <p className="text-xs text-gray-400 mb-1 uppercase tracking-wide">{title}</p>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="step" tick={{ fontSize: 10, fill: '#9ca3af' }} />
          <YAxis tick={{ fontSize: 10, fill: '#9ca3af' }} width={40} />
          <Tooltip
            contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', fontSize: 11 }}
            labelStyle={{ color: '#9ca3af' }}
          />
          <Legend wrapperStyle={{ fontSize: 10 }} />
          {available.map((key, i) => (
            <Line
              key={key} type="monotone" dataKey={key}
              stroke={METRIC_COLORS[i % METRIC_COLORS.length]}
              dot={false} strokeWidth={1.5}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function LogViewer({ log }: { log: string }) {
  const ref = useRef<HTMLPreElement>(null)
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [log])
  const lines = log.split('\n').slice(-80).join('\n')
  return (
    <pre
      ref={ref}
      className="bg-gray-900 rounded p-3 text-xs text-gray-300 h-40 overflow-y-auto font-mono whitespace-pre-wrap"
    >
      {lines || <span className="text-gray-600">No log output yet.</span>}
    </pre>
  )
}

function NewRunDialog({
  onClose, onCreated,
}: {
  onClose: () => void
  onCreated: (id: string) => void
}) {
  const [name, setName] = useState('')
  const [pyConfig, setPyConfig] = useState('config/config.py')
  const [manifestTrain, setManifestTrain] = useState('')
  const [manifestVal, setManifestVal] = useState('')
  const [selectedConfigId, setSelectedConfigId] = useState('')
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const { data: recipes = [] } = useQuery({ queryKey: ['recipes'], queryFn: listRecipes })
  const { data: savedConfigs = [] } = useQuery({ queryKey: ['configs'], queryFn: listConfigs })

  function selectRecipe(recipeId: string) {
    const r = (recipes as any[]).find((x: any) => x.id === recipeId)
    if (r?.train_manifest) setManifestTrain(r.train_manifest)
    if (r?.val_manifest) setManifestVal(r.val_manifest)
  }

  async function handleCreate() {
    if (!name.trim()) { setError('Name required'); return }
    setCreating(true)
    setError(null)
    try {
      const job = await createTrainJob({ name, py_config: pyConfig, manifest_train: manifestTrain, manifest_val: manifestVal })
      onCreated(job.id)
      onClose()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to create job')
    } finally {
      setCreating(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6 w-[480px]">
        <h2 className="text-base font-bold mb-4">New Training Run</h2>
        <div className="flex flex-col gap-3">
          <div>
            <label className="text-xs text-gray-400 block mb-1">Run name *</label>
            <input value={name} onChange={e => setName(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm" />
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Config file</label>
            <input value={pyConfig} onChange={e => { setPyConfig(e.target.value); setSelectedConfigId('') }}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm" />
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">From recipe (optional)</label>
            <select onChange={e => selectRecipe(e.target.value)} defaultValue=""
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm">
              <option value="">— select recipe —</option>
              {(recipes as any[]).map((r: any) => (
                <option key={r.id} value={r.id}>{r.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">From config builder (optional)</label>
            <select
              value={selectedConfigId}
              onChange={async (e) => {
                const id = e.target.value
                setSelectedConfigId(id)
                if (!id) return
                try {
                  const { path } = await writeConfigToDisk(id)
                  setPyConfig(path)
                } catch {
                  setError('Failed to export config to disk')
                }
              }}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm"
            >
              <option value="">— use path above —</option>
              {(savedConfigs as ConfigRecord[]).map((c: ConfigRecord) => (
                <option key={c.id} value={c.id}>{c.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Train manifest (path)</label>
            <input value={manifestTrain} onChange={e => setManifestTrain(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm" />
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Val manifest (path)</label>
            <input value={manifestVal} onChange={e => setManifestVal(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm" />
          </div>
        </div>
        {error && <p className="text-red-400 text-xs mt-2">{error}</p>}
        <div className="flex justify-end gap-2 mt-4">
          <button onClick={onClose} className="px-3 py-1.5 text-sm text-gray-400 hover:text-gray-200">Cancel</button>
          <button onClick={handleCreate} disabled={creating}
            className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm disabled:opacity-50">
            {creating ? 'Creating...' : 'Launch'}
          </button>
        </div>
      </div>
    </div>
  )
}

function RunDetail({ jobId }: { jobId: string }) {
  const isRunning = (status: string) => status === 'running' || status === 'queued'

  const { data: job, refetch: refetchJob } = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId),
    refetchInterval: (query) => isRunning(query.state.data?.status ?? '') ? 3000 : false,
  })

  const { data: metrics = {} } = useQuery({
    queryKey: ['metrics', jobId],
    queryFn: () => getMetrics(jobId),
    refetchInterval: () => {
      // keep polling while running
      return false // will be handled by SSE
    },
    enabled: !!jobId,
  })

  // SSE log streaming
  const [streamedLog, setStreamedLog] = useState('')
  useEffect(() => {
    if (!job || !isRunning(job.status)) return
    const es = new EventSource(`http://localhost:8001/api/jobs/${jobId}/stream`)
    es.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.type === 'log') setStreamedLog(prev => prev + msg.line + '\n')
        if (msg.type === 'status') { refetchJob(); es.close() }
      } catch {}
    }
    return () => es.close()
  }, [jobId, job?.status])

  const [activeTab, setActiveTab] = useState<'loss' | 'val' | 'log'>('loss')

  if (!job) return <p className="text-gray-500">Loading...</p>

  const lossKeys = Object.keys(metrics).filter(k => k.startsWith('train/'))
  const valKeys = Object.keys(metrics).filter(k => k.startsWith('val/'))
  const logContent = streamedLog || job.log || ''

  return (
    <div>
      <div className="flex items-center gap-3 mb-4">
        <h2 className="text-base font-bold">{job.name}</h2>
        <StatusBadge status={job.status} />
        {job.status === 'running' && (
          <button onClick={() => cancelJob(jobId).then(() => refetchJob())}
            className="ml-auto text-xs text-red-400 hover:text-red-300 border border-red-800 rounded px-2 py-0.5">
            Stop
          </button>
        )}
      </div>

      <div className="flex gap-2 mb-4 border-b border-gray-800 pb-2">
        {(['loss', 'val', 'log'] as const).map(tab => (
          <button key={tab} onClick={() => setActiveTab(tab)}
            className={`px-3 py-1 rounded text-sm ${activeTab === tab ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'}`}>
            {tab === 'loss' ? 'Loss' : tab === 'val' ? 'Val Metrics' : 'Log'}
          </button>
        ))}
      </div>

      {activeTab === 'loss' && (
        <MetricsChart metrics={metrics} title="Training Loss" metricKeys={lossKeys} />
      )}
      {activeTab === 'val' && (
        <MetricsChart metrics={metrics} title="Validation Metrics" metricKeys={valKeys} />
      )}
      {activeTab === 'log' && <LogViewer log={logContent} />}

      {job.error && (
        <div className="mt-3 text-xs text-red-400 bg-red-900/20 border border-red-800 rounded p-2">
          {job.error}
        </div>
      )}
    </div>
  )
}

export default function Training() {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [showNew, setShowNew] = useState(false)

  const { data: jobs = [], refetch } = useQuery({
    queryKey: ['jobs'],
    queryFn: listJobs,
    refetchInterval: 5000,
  })

  return (
    <div className="flex gap-6 h-[calc(100vh-5rem)]">
      {/* Left: Job list */}
      <div className="w-60 shrink-0 border-r border-gray-800 pr-4 overflow-y-auto">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-base font-bold">Runs</h2>
          <button onClick={() => setShowNew(true)}
            className="text-xs bg-blue-700 hover:bg-blue-600 text-white rounded px-2 py-1">
            + New
          </button>
        </div>
        {(jobs as Job[]).length === 0 && <p className="text-gray-500 text-sm">No runs yet.</p>}
        {(jobs as Job[]).map(job => (
          <button key={job.id} onClick={() => setSelectedId(job.id)}
            className={`w-full text-left p-2 rounded mb-1 text-sm transition-colors ${selectedId === job.id ? 'bg-gray-800' : 'hover:bg-gray-900'}`}>
            <div className="font-medium truncate">{job.name}</div>
            <div className="flex items-center justify-between mt-0.5">
              <StatusBadge status={job.status} />
              <span className="text-xs text-gray-600">{job.type}</span>
            </div>
          </button>
        ))}
      </div>

      {/* Right: Run detail */}
      <div className="flex-1 overflow-y-auto">
        {selectedId
          ? <RunDetail jobId={selectedId} />
          : <p className="text-gray-500 mt-10">Select a run from the sidebar.</p>
        }
      </div>

      {showNew && (
        <NewRunDialog
          onClose={() => setShowNew(false)}
          onCreated={(id) => { setSelectedId(id); refetch() }}
        />
      )}
    </div>
  )
}
