import { useState, useEffect } from 'react'
import { startSession } from '../api'

export function useSession() {
  const [sessionId, setSessionId]   = useState<string | null>(null)
  const [error, setError]           = useState<string | null>(null)
  const [loading, setLoading]       = useState(true)

  useEffect(() => {
    startSession()
      .then(d => setSessionId(d.session_id))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  return { sessionId, error, loading }
}