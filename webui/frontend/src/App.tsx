import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import SplitManager from './pages/SplitManager'
import Training from './pages/Training'
import Evaluation from './pages/Evaluation'

const queryClient = new QueryClient()

function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <nav className="border-b border-gray-800 px-6 py-3 flex gap-6 items-center">
        <span className="font-bold text-lg tracking-tight">6Img-to-3D</span>
        {[
          { to: '/splits', label: 'Split Manager' },
          { to: '/training', label: 'Training' },
          { to: '/eval', label: 'Evaluation' },
        ].map(({ to, label }) => (
          <NavLink key={to} to={to} className={({ isActive }) =>
            isActive ? 'text-blue-400' : 'text-gray-400 hover:text-gray-200'
          }>{label}</NavLink>
        ))}
      </nav>
      <main className="p-6">{children}</main>
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<SplitManager />} />
            <Route path="/splits" element={<SplitManager />} />
            <Route path="/training" element={<Training />} />
            <Route path="/eval" element={<Evaluation />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
