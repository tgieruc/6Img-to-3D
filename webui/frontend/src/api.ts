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
