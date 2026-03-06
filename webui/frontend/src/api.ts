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

// ── Config Builder ──────────────────────────────────────────────────────────

export interface PIFConfig {
  enabled: boolean
  factor: number
  transforms_path: string
}

export interface EncoderConfig {
  dim: number
  num_heads: number
  num_levels: number
  max_cams: number
  min_cams_train: number
  tpv_h: number
  tpv_w: number
  tpv_z: number
  num_encoder_layers: number
  scene_contraction: boolean
  scene_contraction_factor: number[]
  offset: number[]
  scale: number[]
  num_points_in_pillar: number[]
  num_points: number[]
  hybrid_attn_anchors: number
  hybrid_attn_points: number
}

export interface DecoderConfig {
  hidden_dim: number
  hidden_layers: number
  density_activation: string
  nb_bins: number
  nb_bins_sample: number
  hn: number
  hf: number
  train_stratified: boolean
  white_background: boolean
  whiteout: boolean
  testing_batch_size: number
}

export interface OptimizerConfig {
  lr: number
  num_epochs: number
  num_warmup_steps: number
  lpips_loss_weight: number
  tv_loss_weight: number
  dist_loss_weight: number
  depth_loss_weight: number
  clip_grad_norm: number
}

export interface TrainLoaderConfig {
  pickled: boolean
  batch_size: number
  shuffle: boolean
  num_workers: number
  towns: string[]
  weather: string[]
  vehicle: string[]
  factor: number
  num_imgs: number
  depth: boolean
  min_cams_train: number
  max_cams_train: number
}

export interface ValLoaderConfig {
  pickled: boolean
  phase: string
  batch_size: number
  num_workers: number
  towns: string[]
  weather: string[]
  vehicle: string[]
  spawn_point: number[]
  factor: number
  depth: boolean
}

export interface DatasetConfig {
  data_path: string
  train: TrainLoaderConfig
  val: ValLoaderConfig
}

export interface FullConfig {
  encoder: EncoderConfig
  decoder: DecoderConfig
  optimizer: OptimizerConfig
  dataset: DatasetConfig
  pif: PIFConfig
}

export interface ConfigRecord {
  id: string
  name: string
  created_at: string
  data?: FullConfig
}

export async function listConfigs(): Promise<ConfigRecord[]> {
  const res = await api.get('/api/configs')
  return res.data
}

export async function getConfig(id: string): Promise<ConfigRecord & { data: FullConfig }> {
  const res = await api.get(`/api/configs/${id}`)
  return res.data
}

export async function createConfig(name: string, data: FullConfig): Promise<ConfigRecord> {
  const res = await api.post(`/api/configs?name=${encodeURIComponent(name)}`, data)
  return res.data
}

export async function updateConfig(id: string, name: string, data: FullConfig): Promise<ConfigRecord> {
  const res = await api.put(`/api/configs/${id}?name=${encodeURIComponent(name)}`, data)
  return res.data
}

export async function deleteConfig(id: string): Promise<void> {
  await api.delete(`/api/configs/${id}`)
}

export async function cloneConfig(id: string, newName: string): Promise<ConfigRecord> {
  const res = await api.post(`/api/configs/${id}/clone?new_name=${encodeURIComponent(newName)}`)
  return res.data
}

export async function importConfig(name: string, file: File): Promise<ConfigRecord & { data: FullConfig }> {
  const form = new FormData()
  form.append('file', file)
  const res = await api.post(`/api/configs/import?name=${encodeURIComponent(name)}`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data
}

export function configExportUrl(id: string): string {
  return `http://localhost:8001/api/configs/${id}/export`
}

export async function writeConfigToDisk(id: string): Promise<{ path: string }> {
  const res = await api.post(`/api/configs/${id}/write`)
  return res.data
}
