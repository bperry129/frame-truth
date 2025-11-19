import React, { useEffect, useState } from 'react';
import { getSubmissions } from '../services/api';
import { PlayCircle, Calendar, AlertCircle, CheckCircle, Globe, Lock } from 'lucide-react';

interface HistoryProps {
  adminAuth: { user: string; pass: string };
  onLogout: () => void;
}

const History: React.FC<HistoryProps> = ({ adminAuth, onLogout }) => {
  const [submissions, setSubmissions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const data = await getSubmissions(adminAuth.user, adminAuth.pass);
      setSubmissions(data);
    } catch (e) {
      setError("Access Denied or Server Error");
    }
    setLoading(false);
  };

  const formatDate = (isoString: string) => {
    return new Date(isoString).toLocaleString();
  };

  const viewSubmission = (id: string) => {
      // Full refresh to reset app state properly or navigation if we had router
      window.location.href = `/?id=${id}`;
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-12 animate-fade-in">
       <div className="mb-12 text-center relative">
        <button onClick={onLogout} className="absolute right-0 top-0 text-xs text-slate-500 hover:text-white flex items-center gap-1">
            <Lock className="w-3 h-3" /> Logout
        </button>
        <h2 className="text-3xl font-bold text-white mb-4">Submission <span className="text-cyan-400">History</span></h2>
        <p className="text-slate-400">Restricted Admin Area</p>
      </div>

      {loading ? (
          <div className="text-center text-slate-500 py-20">Authenticating...</div>
      ) : error ? (
          <div className="text-center text-rose-500 py-20 border border-rose-900/30 rounded-xl bg-rose-500/5">
              <AlertCircle className="w-8 h-8 mx-auto mb-2" />
              {error}
          </div>
      ) : submissions.length === 0 ? (
          <div className="text-center text-slate-500 py-20 border border-slate-800 rounded-xl bg-slate-900/50">No submissions found.</div>
      ) : (
          <div className="grid gap-4">
              {submissions.map((sub) => (
                  <div key={sub.id} onClick={() => viewSubmission(sub.id)} className="bg-slate-900/50 border border-slate-800 p-4 rounded-xl flex flex-col md:flex-row items-center gap-6 hover:bg-slate-900 hover:border-cyan-500/30 transition-all cursor-pointer group">
                      {/* Thumbnail Placeholder / Video Preview */}
                      <div className="w-full md:w-40 h-24 bg-black rounded-lg overflow-hidden relative border border-slate-800 group-hover:border-cyan-500/50 flex-shrink-0">
                          <video src={sub.video_url} className="w-full h-full object-cover opacity-60" />
                          <div className="absolute inset-0 flex items-center justify-center bg-black/20">
                             <PlayCircle className="w-8 h-8 text-white/80" />
                          </div>
                      </div>
                      
                      <div className="flex-1 w-full">
                          <div className="flex items-start justify-between mb-2">
                              <div className="flex-1 min-w-0 mr-4">
                                  <span className="text-xs font-mono text-slate-500 uppercase tracking-wider">ID: {sub.id}</span>
                                  {sub.original_url && sub.original_url.startsWith('http') ? (
                                      <a 
                                        href={sub.original_url} 
                                        target="_blank" 
                                        rel="noreferrer"
                                        className="block text-lg font-medium text-cyan-400 hover:underline break-all"
                                        onClick={(e) => e.stopPropagation()}
                                      >
                                          {sub.original_url}
                                      </a>
                                  ) : (
                                      <h3 className="text-lg font-medium text-white truncate max-w-md" title={sub.original_url}>
                                          {sub.original_url || "Uploaded Local File"}
                                      </h3>
                                  )}
                              </div>
                              
                              {/* Verdict Badge */}
                              {sub.summary.isAi ? (
                                  <div className="flex items-center gap-1.5 bg-rose-500/10 text-rose-400 px-3 py-1 rounded-full text-xs font-bold border border-rose-500/20">
                                      <AlertCircle className="w-3 h-3" /> AI Generated
                                  </div>
                              ) : (
                                   <div className="flex items-center gap-1.5 bg-emerald-500/10 text-emerald-400 px-3 py-1 rounded-full text-xs font-bold border border-emerald-500/20">
                                      <CheckCircle className="w-3 h-3" /> Real Footage
                                  </div>
                              )}
                          </div>
                          
                          <div className="flex flex-wrap gap-4 text-sm text-slate-400">
                              <div className="flex items-center gap-1.5">
                                  <Calendar className="w-3 h-3" />
                                  {formatDate(sub.created_at)}
                              </div>
                              <div className="flex items-center gap-1.5 text-slate-500" title="Submitter IP">
                                  <Globe className="w-3 h-3" />
                                  {sub.ip_address || "Unknown IP"}
                              </div>
                              {sub.summary.modelDetected && (
                                   <div>
                                       Generator: <span className="text-cyan-500">{sub.summary.modelDetected}</span>
                                   </div>
                              )}
                               {sub.summary.confidence && (
                                   <div>
                                       Confidence: <span className="text-white">{sub.summary.confidence}%</span>
                                   </div>
                              )}
                          </div>
                      </div>
                  </div>
              ))}
          </div>
      )}
    </div>
  );
};

export default History;
