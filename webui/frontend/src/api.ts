import axios from 'axios'

const api = axios.create({ baseURL: 'http://localhost:8001' })

export interface DataOptions {
  towns: string[]
  weathers: string[]
  vehicles: string[]
  sensors: string[]
}

export interface SplitRule {
  towns: string[]
  spawn_points: string | number[]
  steps: string | number[]
}

export interface RecipePayload {
  name: string
  data_dir: string
  output_dir: string
  global_filters: {
    vehicles: string[]
    weathers: string[]
    input_sensor: string
    target_sensor: string
  }
  splits: Record<string, SplitRule>
}

export async function getDataOptions(): Promise<DataOptions> {
  const res = await api.get('/api/data/options')
  return res.data
}

export async function getDataScan() {
  const res = await api.get('/api/data/scan')
  return res.data
}

export async function previewSplit(payload: {
  data_dir: string
  global_filters: RecipePayload['global_filters']
  splits: Record<string, SplitRule>
}): Promise<Record<string, number>> {
  const res = await api.post('/api/data/preview', payload)
  return res.data
}

export async function createRecipe(payload: RecipePayload) {
  const res = await api.post('/api/recipes', payload)
  return res.data
}

export async function exportRecipe(id: string) {
  const res = await api.post(`/api/recipes/${id}/export`)
  return res.data
}

export async function listRecipes() {
  const res = await api.get('/api/recipes')
  return res.data
}

export interface Job {
  id: string
  name: string
  type: string
  status: string
  mlflow_run_id: string | null
  created_at: string | null
  started_at: string | null
  completed_at: string | null
  error: string | null
}

export interface MetricPoint { step: number; value: number }

export async function listJobs(): Promise<Job[]> {
  const res = await api.get('/api/jobs')
  return res.data
}

export async function getJob(id: string): Promise<Job & { log: string }> {
  const res = await api.get(`/api/jobs/${id}`)
  return res.data
}

export async function getMetrics(id: string): Promise<Record<string, MetricPoint[]>> {
  const res = await api.get(`/api/jobs/${id}/metrics`)
  return res.data
}

export async function createTrainJob(payload: {
  name: string
  py_config: string
  manifest_train: string
  manifest_val: string
}): Promise<{ id: string; status: string }> {
  const res = await api.post('/api/jobs/train', payload)
  return res.data
}

export async function cancelJob(id: string): Promise<void> {
  await api.delete(`/api/jobs/${id}`)
}

export async function createEvalJob(payload: {
  name: string
  resume_from: string
  manifest_val: string
  py_config: string
}): Promise<{ id: string; status: string }> {
  const res = await api.post('/api/jobs/eval', payload)
  return res.data
}

export async function listRenders(jobId: string): Promise<string[]> {
  const res = await api.get(`/api/jobs/${jobId}/renders`)
  return res.data
}

export function renderUrl(jobId: string, filename: string): string {
  return `http://localhost:8001/api/jobs/${jobId}/renders/${filename}`
}
