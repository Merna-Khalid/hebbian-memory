import { useState, useEffect } from 'react'
import { startSecondBrainSession } from '../api'

export function useSecondBrainSession() {
  const [sessionId, setSessionId]   = useState<string | null>(null)
  const [error, setError]           = useState<string | null>(null)
  const [loading, setLoading]       = useState(true)

  useEffect(() => {
    startSecondBrainSession()
      .then(d => setSessionId(d.session_id))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  return { sessionId, error, loading }
}
