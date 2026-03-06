import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getDataOptions, previewSplit, createRecipe, exportRecipe } from '../api'

const DATA_DIR = '/home/bdw/Documents/seed4d/data'
const OUTPUT_DIR = '/home/bdw/Documents/seed4d'

const SPLITS = ['train', 'val', 'test'] as const
type SplitName = typeof SPLITS[number]

const SPLIT_COLORS: Record<SplitName, string> = {
  train: 'text-blue-400',
  val: 'text-amber-400',
  test: 'text-green-400',
}

const SPLIT_BG: Record<SplitName, string> = {
  train: 'border-blue-700 bg-blue-900/20',
  val: 'border-amber-700 bg-amber-900/20',
  test: 'border-green-700 bg-green-900/20',
}

export default function SplitManager() {
  const { data: options } = useQuery({ queryKey: ['data-options'], queryFn: getDataOptions })

  const [vehicles, setVehicles] = useState<string[]>([])
  const [weathers, setWeathers] = useState<string[]>([])
  const [inputSensor, setInputSensor] = useState('nuscenes')
  const [targetSensor, setTargetSensor] = useState('sphere')

  const [splitTowns, setSplitTowns] = useState<Record<SplitName, string[]>>({
    train: [], val: [], test: [],
  })

  const [preview, setPreview] = useState<Record<string, number>>({})
  const [recipeName, setRecipeName] = useState('my-split')
  const [status, setStatus] = useState<string | null>(null)

  // Live preview: debounced, fires when any filter changes
  useEffect(() => {
    if (!options) return
    const timer = setTimeout(async () => {
      try {
        const counts = await previewSplit({
          data_dir: DATA_DIR,
          global_filters: { vehicles, weathers, input_sensor: inputSensor, target_sensor: targetSensor },
          splits: {
            train: { towns: splitTowns.train, spawn_points: 'all', steps: 'all' },
            val: { towns: splitTowns.val, spawn_points: 'all', steps: 'all' },
            test: { towns: splitTowns.test, spawn_points: 'all', steps: 'all' },
          },
        })
        setPreview(counts)
      } catch (e) {
        // backend may not be running yet
      }
    }, 400)
    return () => clearTimeout(timer)
  }, [vehicles, weathers, inputSensor, targetSensor, splitTowns, options])

  // Which towns are already assigned?
  const assignedTowns = new Set(Object.values(splitTowns).flat())

  function toggleTown(split: SplitName, town: string) {
    setSplitTowns(prev => {
      const current = prev[split]
      if (current.includes(town)) {
        return { ...prev, [split]: current.filter(t => t !== town) }
      } else {
        // Remove from other splits first (mutual exclusivity)
        const newState = { ...prev }
        for (const s of SPLITS) {
          if (s !== split) newState[s] = newState[s].filter(t => t !== town)
        }
        return { ...newState, [split]: [...current, town] }
      }
    })
  }

  async function handleExport() {
    setStatus('Creating recipe...')
    try {
      const payload = {
        name: recipeName,
        data_dir: DATA_DIR,
        output_dir: OUTPUT_DIR,
        global_filters: { vehicles, weathers, input_sensor: inputSensor, target_sensor: targetSensor },
        splits: {
          train: { towns: splitTowns.train, spawn_points: 'all', steps: 'all' },
          val: { towns: splitTowns.val, spawn_points: 'all', steps: 'all' },
          test: { towns: splitTowns.test, spawn_points: 'all', steps: 'all' },
        },
      }
      const recipe = await createRecipe(payload)
      setStatus('Exporting manifests...')
      const result = await exportRecipe(recipe.id)
      setStatus(`Done! train: ${result.scene_counts?.train ?? 0}, val: ${result.scene_counts?.val ?? 0}, test: ${result.scene_counts?.test ?? 0} scenes`)
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } }; message?: string }
      setStatus(`Error: ${err?.response?.data?.detail ?? err?.message ?? 'unknown'}`)
    }
  }

  const allTowns = options?.towns ?? []
  const allVehicles = options?.vehicles ?? []
  const allWeathers = options?.weathers ?? []
  const allSensors = options?.sensors ?? []

  return (
    <div className="flex gap-6 h-[calc(100vh-5rem)]">
      {/* Left: Global Filters */}
      <div className="w-56 shrink-0 border-r border-gray-800 pr-4 overflow-y-auto">
        <h2 className="text-base font-bold mb-4">Global Filters</h2>

        <FilterSection label="Vehicles" items={allVehicles} selected={vehicles} onToggle={v =>
          setVehicles(vs => vs.includes(v) ? vs.filter(x => x !== v) : [...vs, v])
        } />

        <FilterSection label="Weathers" items={allWeathers} selected={weathers} onToggle={w =>
          setWeathers(ws => ws.includes(w) ? ws.filter(x => x !== w) : [...ws, w])
        } />

        <div className="mb-4">
          <p className="text-xs text-gray-400 mb-1">Input sensor</p>
          <select value={inputSensor} onChange={e => setInputSensor(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm">
            {allSensors.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>

        <div className="mb-4">
          <p className="text-xs text-gray-400 mb-1">Target sensor</p>
          <select value={targetSensor} onChange={e => setTargetSensor(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm">
            {allSensors.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
      </div>

      {/* Center: Split Rules */}
      <div className="flex-1 overflow-y-auto">
        <h2 className="text-base font-bold mb-4">Split Assignments</h2>
        <div className="flex flex-col gap-4">
          {SPLITS.map(split => (
            <div key={split} className={`border rounded-lg p-4 ${SPLIT_BG[split]}`}>
              <h3 className={`text-sm font-semibold mb-3 uppercase tracking-wide ${SPLIT_COLORS[split]}`}>
                {split}
                <span className="text-gray-500 font-normal ml-2 text-xs">
                  {splitTowns[split].length} town{splitTowns[split].length !== 1 ? 's' : ''}
                </span>
              </h3>
              <div className="flex flex-wrap gap-2">
                {allTowns.length === 0 && <p className="text-xs text-gray-500">Loading...</p>}
                {allTowns.map(town => {
                  const inThis = splitTowns[split].includes(town)
                  const inOther = !inThis && assignedTowns.has(town)
                  return (
                    <button
                      key={town}
                      onClick={() => !inOther && toggleTown(split, town)}
                      disabled={inOther}
                      className={`px-2.5 py-1 rounded text-xs border transition-colors ${
                        inThis
                          ? `${SPLIT_COLORS[split]} border-current bg-current/10 font-medium`
                          : inOther
                          ? 'text-gray-600 border-gray-700 cursor-not-allowed opacity-40'
                          : 'text-gray-300 border-gray-600 hover:border-gray-400'
                      }`}
                    >
                      {town}
                    </button>
                  )
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Right: Preview + Export */}
      <div className="w-52 shrink-0 border-l border-gray-800 pl-4 flex flex-col">
        <h2 className="text-base font-bold mb-4">Preview</h2>
        <div className="flex flex-col gap-2 mb-6">
          {SPLITS.map(split => (
            <div key={split} className="flex justify-between items-center text-sm">
              <span className={`${SPLIT_COLORS[split]} uppercase text-xs font-medium`}>{split}</span>
              <span className="text-gray-300 font-mono">{preview[split] ?? '—'}</span>
            </div>
          ))}
          <p className="text-xs text-gray-500 mt-1">scenes</p>
        </div>

        <div className="mb-3">
          <label className="text-xs text-gray-400 block mb-1">Recipe name</label>
          <input value={recipeName} onChange={e => setRecipeName(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm" />
        </div>

        <button
          onClick={handleExport}
          className="w-full bg-blue-600 hover:bg-blue-500 text-white rounded px-3 py-2 text-sm font-medium transition-colors"
        >
          Export Manifests
        </button>

        {status && (
          <p className="mt-3 text-xs text-gray-400 break-words">{status}</p>
        )}
      </div>
    </div>
  )
}

function FilterSection({ label, items, selected, onToggle }: {
  label: string
  items: string[]
  selected: string[]
  onToggle: (item: string) => void
}) {
  return (
    <div className="mb-4">
      <p className="text-xs text-gray-400 mb-1">{label}</p>
      <div className="flex flex-col gap-1">
        {items.map(item => (
          <label key={item} className="flex items-center gap-2 text-xs cursor-pointer">
            <input type="checkbox" checked={selected.includes(item)} onChange={() => onToggle(item)}
              className="accent-blue-500" />
            <span className="text-gray-300 truncate" title={item}>{item}</span>
          </label>
        ))}
      </div>
    </div>
  )
}
