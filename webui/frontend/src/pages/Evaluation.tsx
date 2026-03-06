import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { listJobs, getJob, listRenders, createEvalJob, renderUrl, listRecipes, type Job } from '../api'

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    queued: 'text-gray-400', running: 'text-blue-400',
    completed: 'text-green-400', failed: 'text-red-400', cancelled: 'text-yellow-400',
  }
  const dots: Record<string, string> = {
    queued: 'bg-gray-500', running: 'bg-blue-400 animate-pulse',
    completed: 'bg-green-400', failed: 'bg-red-400', cancelled: 'bg-yellow-400',
  }
  return (
    <span className={`flex items-center gap-1.5 text-xs ${colors[status] ?? 'text-gray-400'}`}>
      <span className={`w-1.5 h-1.5 rounded-full inline-block ${dots[status] ?? 'bg-gray-500'}`} />
      {status}
    </span>
  )
}

function RunEvalDialog({
  onClose, onCreated,
}: {
  onClose: () => void
  onCreated: (id: string) => void
}) {
  const [name, setName] = useState('')
  const [resumeFrom, setResumeFrom] = useState('')
  const [manifestVal, setManifestVal] = useState('')
  const [pyConfig, setPyConfig] = useState('config/config.py')
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const { data: recipes = [] } = useQuery({ queryKey: ['recipes'], queryFn: listRecipes })

  function selectRecipe(recipeId: string) {
    const r = (recipes as any[]).find((x: any) => x.id === recipeId)
    if (r?.val_manifest) setManifestVal(r.val_manifest)
  }

  async function handleCreate() {
    if (!name.trim() || !resumeFrom.trim()) {
      setError('Name and checkpoint path are required')
      return
    }
    setCreating(true)
    setError(null)
    try {
      const job = await createEvalJob({ name, resume_from: resumeFrom, manifest_val: manifestVal, py_config: pyConfig })
      onCreated(job.id)
      onClose()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to create eval job')
    } finally {
      setCreating(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6 w-[480px]">
        <h2 className="text-base font-bold mb-4">Run Evaluation</h2>
        <div className="flex flex-col gap-3">
          <div>
            <label className="text-xs text-gray-400 block mb-1">Run name *</label>
            <input value={name} onChange={e => setName(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm" />
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Checkpoint path *</label>
            <input value={resumeFrom} onChange={e => setResumeFrom(e.target.value)}
              placeholder="runs/.../models/model_best_psnr.pth"
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm" />
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Config file</label>
            <input value={pyConfig} onChange={e => setPyConfig(e.target.value)}
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
            <label className="text-xs text-gray-400 block mb-1">Val manifest path</label>
            <input value={manifestVal} onChange={e => setManifestVal(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm" />
          </div>
        </div>
        {error && <p className="text-red-400 text-xs mt-2">{error}</p>}
        <div className="flex justify-end gap-2 mt-4">
          <button onClick={onClose} className="px-3 py-1.5 text-sm text-gray-400 hover:text-gray-200">Cancel</button>
          <button onClick={handleCreate} disabled={creating}
            className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm disabled:opacity-50">
            {creating ? 'Launching...' : 'Run Eval'}
          </button>
        </div>
      </div>
    </div>
  )
}

function MetricsSummary({ log }: { log: string }) {
  // Parse PSNR/LPIPS/SSIM from eval log output
  const psnrMatch = log.match(/PSNR[:\s]+([0-9.]+)/i)
  const lpipsMatch = log.match(/LPIPS[:\s]+([0-9.]+)/i)
  const ssimMatch = log.match(/SSIM[:\s]+([0-9.]+)/i)

  if (!psnrMatch && !lpipsMatch && !ssimMatch) return null

  return (
    <div className="flex gap-6 mb-4 p-3 bg-gray-900 rounded border border-gray-700">
      {psnrMatch && (
        <div className="text-center">
          <p className="text-xs text-gray-400">PSNR</p>
          <p className="text-lg font-mono font-bold text-blue-400">{parseFloat(psnrMatch[1]).toFixed(2)}</p>
          <p className="text-xs text-gray-500">dB</p>
        </div>
      )}
      {lpipsMatch && (
        <div className="text-center">
          <p className="text-xs text-gray-400">LPIPS</p>
          <p className="text-lg font-mono font-bold text-amber-400">{parseFloat(lpipsMatch[1]).toFixed(3)}</p>
        </div>
      )}
      {ssimMatch && (
        <div className="text-center">
          <p className="text-xs text-gray-400">SSIM</p>
          <p className="text-lg font-mono font-bold text-green-400">{parseFloat(ssimMatch[1]).toFixed(3)}</p>
        </div>
      )}
    </div>
  )
}

function RenderGallery({ jobId }: { jobId: string }) {
  const { data: renders = [] } = useQuery({
    queryKey: ['renders', jobId],
    queryFn: () => listRenders(jobId),
    refetchInterval: 5000,
  })

  if (renders.length === 0) {
    return <p className="text-gray-500 text-sm">No rendered images yet. Run eval to generate them.</p>
  }

  return (
    <div className="grid grid-cols-3 gap-3">
      {(renders as string[]).map(filename => (
        <div key={filename} className="border border-gray-700 rounded overflow-hidden">
          <img
            src={renderUrl(jobId, filename)}
            alt={filename}
            className="w-full bg-gray-900"
            loading="lazy"
          />
          <p className="text-xs text-gray-500 p-1.5 truncate">{filename}</p>
        </div>
      ))}
    </div>
  )
}

function EvalDetail({ jobId }: { jobId: string }) {
  const { data: job } = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId),
    refetchInterval: (q) => {
      const s = q.state.data?.status ?? ''
      return s === 'running' || s === 'queued' ? 3000 : false
    },
  })

  const [activeTab, setActiveTab] = useState<'renders' | 'log'>('renders')

  if (!job) return <p className="text-gray-500">Loading...</p>

  return (
    <div>
      <div className="flex items-center gap-3 mb-4">
        <h2 className="text-base font-bold">{job.name}</h2>
        <StatusBadge status={job.status} />
      </div>

      {job.log && <MetricsSummary log={job.log} />}

      <div className="flex gap-2 mb-4 border-b border-gray-800 pb-2">
        {(['renders', 'log'] as const).map(tab => (
          <button key={tab} onClick={() => setActiveTab(tab)}
            className={`px-3 py-1 rounded text-sm ${activeTab === tab ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'}`}>
            {tab === 'renders' ? 'Renders' : 'Log'}
          </button>
        ))}
      </div>

      {activeTab === 'renders' && <RenderGallery jobId={jobId} />}
      {activeTab === 'log' && (
        <pre className="bg-gray-900 rounded p-3 text-xs text-gray-300 h-80 overflow-y-auto font-mono whitespace-pre-wrap">
          {job.log || <span className="text-gray-600">No log output yet.</span>}
        </pre>
      )}

      {job.error && (
        <div className="mt-3 text-xs text-red-400 bg-red-900/20 border border-red-800 rounded p-2">
          {job.error}
        </div>
      )}
    </div>
  )
}

export default function Evaluation() {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [showNew, setShowNew] = useState(false)

  const { data: jobs = [], refetch } = useQuery({
    queryKey: ['jobs'],
    queryFn: listJobs,
    refetchInterval: 8000,
    select: (data) => (data as Job[]).filter(j => j.type === 'eval'),
  })

  return (
    <div className="flex gap-6 h-[calc(100vh-5rem)]">
      {/* Left: eval job list */}
      <div className="w-60 shrink-0 border-r border-gray-800 pr-4 overflow-y-auto">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-base font-bold">Eval Runs</h2>
          <button onClick={() => setShowNew(true)}
            className="text-xs bg-blue-700 hover:bg-blue-600 text-white rounded px-2 py-1">
            + Run Eval
          </button>
        </div>
        {(jobs as Job[]).length === 0 && <p className="text-gray-500 text-sm">No eval runs yet.</p>}
        {(jobs as Job[]).map(job => (
          <button key={job.id} onClick={() => setSelectedId(job.id)}
            className={`w-full text-left p-2 rounded mb-1 text-sm transition-colors ${selectedId === job.id ? 'bg-gray-800' : 'hover:bg-gray-900'}`}>
            <div className="font-medium truncate">{job.name}</div>
            <div className="mt-0.5">
              <StatusBadge status={job.status} />
            </div>
          </button>
        ))}
      </div>

      {/* Right: eval detail */}
      <div className="flex-1 overflow-y-auto">
        {selectedId
          ? <EvalDetail jobId={selectedId} />
          : <p className="text-gray-500 mt-10">Select an eval run from the sidebar, or click "+ Run Eval" to start one.</p>
        }
      </div>

      {showNew && (
        <RunEvalDialog
          onClose={() => setShowNew(false)}
          onCreated={(id) => { setSelectedId(id); refetch() }}
        />
      )}
    </div>
  )
}
